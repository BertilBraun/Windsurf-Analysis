"""Debug video renderer (temporal, current vs. previous frame).

User-driven features:
- Frames scaled 0.5× and stacked vertically: **current** frame on top, **previous** frame below.
- Bounding boxes drawn (white = current, light gray = previous).
- All-to-all connector lines current→previous.
- Metrics per connector: cosine similarity, pixel center distance, IoU
- Stable color cycling for connector lines from a 30‑color seeded palette.
- **Collision-aware label placement with per-color cached preferred position:**
  - Each palette color has a cached fractional line position `t` (init 0.25).
  - When drawing a line whose color index = `idx`, attempt to place the text at cached `t_cache[idx]`.
  - If occupied, probe offsets ±0.1, ±0.2, ... alternating up/down until both directions go out of [0,1].
  - First in-bounds, non-overlapping candidate wins; cache updated to that `t`.
  - If no candidate fits, fall back to a random in-bounds `t` (cache unchanged).
  - Label text color matches the connector line color.
- All label rectangles (boxes + metrics text) tracked per output frame to avoid overlaps.

Integrate by importing this file or copying the `_generate_debug_video_worker_function` into your pipeline.

"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from common_types import Detection, BoundingBox, cosine_similarity

# Fixed box colors (BGR)
BOX_COLOR_CUR = (255, 255, 255)  # white
BOX_COLOR_PREV = (170, 170, 170)  # light gray


def _generate_palette(n: int = 30, seed: int = 0) -> List[Tuple[int, int, int]]:
    """Generate a reproducible list of bright, high-contrast BGR colors.

    Uses evenly spaced hues in HSV for good spread; converts to BGR.
    """
    colors: List[Tuple[int, int, int]] = []
    for i in range(n):
        h = int((i / n) * 179)  # OpenCV HSV hue range [0,179]
        s = 200  # high saturation
        v = 255  # max value (brightness)
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    # Shuffle for "random" appearance but deterministic given seed
    rng = np.random.default_rng(seed)
    rng.shuffle(colors)
    return colors


# ---------------------------------------------------------------------------
# Text drawing + collision mgmt ----------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _measure_text(text: str, font_scale: float = 0.4, thickness: int = 1) -> Tuple[int, int, int]:
    (size, base) = cv2.getTextSize(text, _FONT, font_scale, thickness)
    w, h = size
    return w, h, base


def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], margin: int = 2) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1 -= margin
    ay1 -= margin
    ax2 += margin
    ay2 += margin
    bx1 -= margin
    by1 -= margin
    bx2 += margin
    by2 += margin
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _draw_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color=(255, 255, 255),
    font_scale: float = 0.4,
    thickness: int = 1,
    bg: bool = True,
) -> Tuple[int, int, int, int]:
    """Draw text and return bounding rect (x1,y1,x2,y2)."""
    tw, th, base = _measure_text(text, font_scale, thickness)
    x, y = org
    # top-left corner of text box
    tl = (x, y - th - base)
    br = (x + tw, y + base)
    if bg:
        cv2.rectangle(img, tl, br, (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)
    return (tl[0], tl[1], br[0], br[1])


# ---------------------------------------------------------------------------
# Box drawing (records label rect if provided) --------------------------------


def _draw_box(
    img: np.ndarray,
    box: BoundingBox,
    color: Tuple[int, int, int],
    label: Optional[str],
    label_positions: List[Tuple[int, int, int, int]],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    y_offset: int = 0,
) -> Tuple[int, int]:
    x1 = int(box.x1 * scale_x)
    y1 = int(box.y1 * scale_y) + y_offset
    x2 = int(box.x2 * scale_x)
    y2 = int(box.y2 * scale_y) + y_offset
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    if label:
        rect = _draw_text(img, label, (x1, max(y1 - 2, 0)), color)
        label_positions.append(rect)
    return x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2  # center (scaled)


# ---------------------------------------------------------------------------
# Line annotation (metrics) w/ cached + collision-aware placement -------------


def _point_along_line(p1: Tuple[int, int], p2: Tuple[int, int], t: float) -> Tuple[int, int]:
    return (int(round(p1[0] + t * (p2[0] - p1[0]))), int(round(p1[1] + t * (p2[1] - p1[1]))))


def _clamp_label_org(
    img: np.ndarray, text: str, x: int, y: int, font_scale: float, thickness: int
) -> Tuple[int, int, int, int, Tuple[int, int]]:
    """Given a desired anchor (x,y) ~ label center, clamp & return rect + org."""
    h, w = img.shape[:2]
    tw, th, base = _measure_text(text, font_scale, thickness)
    org_x = int(round(x - tw / 2))
    org_y = int(round(y + th / 2))  # baseline near candidate y
    # Clamp to bounds so full box visible
    org_x = max(0, min(org_x, w - tw))
    org_y = max(th + base, min(org_y, h - 1))
    rect = (org_x, org_y - th - base, org_x + tw, org_y + base)
    return rect[0], rect[1], rect[2], rect[3], (org_x, org_y)


def _choose_label_org_for_line_cached(
    img: np.ndarray,
    text: str,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    existing: List[Tuple[int, int, int, int]],
    base_t: float,
    rng: np.random.Generator,
    font_scale: float = 0.35,
    thickness: int = 1,
    step: float = 0.1,
) -> Tuple[Tuple[int, int], float]:
    """Choose label origin along line using cached t + expanding ±step search.

    Returns (org, chosen_t). Does *not* mutate cache; caller updates.
    """
    # Build candidate list: base, +step, -step, +2*step, -2*step, ... until both OOB
    candidates: List[float] = []
    if 0.0 <= base_t <= 1.0:
        candidates.append(base_t)
    k = 1
    upper_done = False
    lower_done = False
    while not (upper_done and lower_done):
        t_up = base_t + step * k
        t_dn = base_t - step * k
        if t_up > 1.0:
            upper_done = True
        else:
            candidates.append(t_up)
        if t_dn < 0.0:
            lower_done = True
        else:
            candidates.append(t_dn)
        k += 1
        # safety: bail if silly loop (shouldn't happen)
        if k > 20:  # step=0.1 => covers >2x range
            break

    # Test candidates
    for t in candidates:
        cand_x, cand_y = _point_along_line(p1, p2, t)
        rx1, ry1, rx2, ry2, org = _clamp_label_org(img, text, cand_x, cand_y, font_scale, thickness)
        rect = (rx1, ry1, rx2, ry2)
        if not any(_rects_overlap(rect, e) for e in existing):
            return org, t

    # fallback random anywhere on line (ignore collisions)
    t = float(rng.random())
    cand_x, cand_y = _point_along_line(p1, p2, t)
    _, _, _, _, org = _clamp_label_org(img, text, cand_x, cand_y, font_scale, thickness)
    return org, t


def _draw_line_with_metrics(
    img: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    cos: float | None,
    dist: float | None,
    iou_dist: float | None,
    color: Tuple[int, int, int],
    label_positions: List[Tuple[int, int, int, int]],
    rng: np.random.Generator,
    *,
    base_t: float,
    cache_update_cb,
) -> float:
    """Draw connector line + metrics label.

    Parameters
    ----------
    base_t:
        Cached preferred fractional position along the line.
    cache_update_cb:
        Callable(new_t: float) -> None used by caller to update per-color cache.

    Returns
    -------
    new_t used for placement (same as passed to callback).
    """
    cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
    cos_s = f'{cos:.2f}' if cos is not None and not math.isnan(cos) else 'n/a'
    dist_s = f'{dist:.0f}' if dist is not None and not math.isnan(dist) else 'n/a'
    iou_s = f'{iou_dist:.2f}' if iou_dist is not None and not math.isnan(iou_dist) else 'n/a'
    txt = f'c={cos_s} iou={iou_s} d={dist_s}'
    org, new_t = _choose_label_org_for_line_cached(img, txt, p1, p2, label_positions, base_t, rng, font_scale=0.35)
    rect = _draw_text(img, txt, org, color, font_scale=0.35)
    label_positions.append(rect)
    cache_update_cb(new_t)
    return new_t


# ---------------------------------------------------------------------------
# Main worker -----------------------------------------------------------------


def generate_debug_video_worker_function(args: tuple[list[Detection], os.PathLike, os.PathLike]) -> None:
    """Multiprocessing worker to render a debug video.

    Parameters
    ----------
    args:
        (det_map, input_path, output_dir)
        det_map: dict mapping frame_index -> list[Detection].
        input_path: source video.
        output_dir: directory to write `<stem>+00_debug.mp4`.
    """
    detections, input_path, output_dir = args
    det_map = defaultdict(list)
    for det in detections:
        det_map[det.frame_idx].append(det)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_video_path = output_dir / f'{input_path.stem}+00_debug.mp4'

    # Lazy imports of project-specific reader/writer
    from video_io import VideoReader, VideoWriter  # type: ignore

    # Stable palette + RNG (seed from video path for reproducibility across runs)
    seed = abs(hash(str(input_path)))
    palette = _generate_palette(30, seed)
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)

    # Per-color cached label fractional positions (init 0.25)
    label_t_cache = [0.25 for _ in range(len(palette))]

    with VideoReader(input_path) as reader:
        video_props = reader.get_properties()  # expects width, height, fps, total_frames

        # debug frame dims: half-width stacked vertically -> same height as original
        scaled_w = max(1, video_props.width // 2)
        scaled_h = max(1, video_props.height // 2)
        debug_w = scaled_w
        debug_h = scaled_h * 2  # top + bottom

        with VideoWriter(annotated_video_path, debug_w, debug_h, video_props.fps) as writer:
            prev_frame_scaled = None  # type: Optional[np.ndarray]
            prev_dets: list[Detection] = []
            prev_idx: Optional[int] = None
            line_counter = 0  # increments per drawn connector line (drives color + cache index)

            for frame_index, frame in tqdm(
                reader.read_frames(), total=video_props.total_frames, desc='Drawing annotations'
            ):
                # Reset label collision registry for this output frame
                label_positions: List[Tuple[int, int, int, int]] = []

                # Resize current frame
                cur_scaled = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

                # Bottom pane: scaled previous or black
                if prev_frame_scaled is not None:
                    bottom = prev_frame_scaled.copy()
                else:
                    bottom = np.zeros_like(cur_scaled)

                # Compose debug canvas (top current, bottom prev)
                canvas = np.zeros((debug_h, debug_w, 3), dtype=cur_scaled.dtype)
                canvas[0:scaled_h, :, :] = cur_scaled
                canvas[scaled_h : scaled_h * 2, :, :] = bottom

                # Draw frame indices
                rect = _draw_text(canvas, f't={frame_index}', (5, 15), (0, 255, 0), bg=True)
                label_positions.append(rect)
                if prev_idx is not None:
                    rect = _draw_text(canvas, f't={prev_idx}', (5, scaled_h + 15), (0, 255, 0), bg=True)
                    label_positions.append(rect)

                # Get detections for current/prev frames
                cur_dets = det_map.get(frame_index, [])
                last_dets = prev_dets if prev_frame_scaled is not None else []

                # Draw current boxes (top)
                cur_centers: list[Tuple[int, int]] = []
                for det in cur_dets:
                    c = _draw_box(
                        canvas, det.bbox, BOX_COLOR_CUR, None, label_positions, scale_x=0.5, scale_y=0.5, y_offset=0
                    )
                    cur_centers.append(c)

                # Draw prev boxes (bottom)
                prev_centers: list[Tuple[int, int]] = []
                for det in last_dets:
                    c = _draw_box(
                        canvas,
                        det.bbox,
                        BOX_COLOR_PREV,
                        None,
                        label_positions,
                        scale_x=0.5,
                        scale_y=0.5,
                        y_offset=scaled_h,
                    )
                    prev_centers.append(c)

                # Pairwise metrics lines current(top) -> prev(bottom)
                if cur_dets and last_dets:
                    for i, det_cur in enumerate(cur_dets):
                        emb_c = det_cur.feat
                        bc = det_cur.bbox
                        c_center = cur_centers[i]
                        for j, det_prev in enumerate(last_dets):
                            emb_p = det_prev.feat
                            bp = det_prev.bbox
                            p_center = prev_centers[j]

                            # metrics
                            cos = (
                                cosine_similarity(emb_c, emb_p)
                                if (emb_c is not None and emb_p is not None)
                                else float('nan')
                            )
                            cx1, cy1 = bc.center
                            cx2, cy2 = bp.center
                            dist = math.hypot(cx2 - cx1, cy2 - cy1)
                            iou = bc.iou(bp)

                            # color + draw
                            color_idx = line_counter % len(palette)
                            color = palette[color_idx]
                            base_t = label_t_cache[color_idx]

                            def _upd(new_t: float, idx=color_idx):
                                label_t_cache[idx] = new_t

                            _draw_line_with_metrics(
                                canvas,
                                c_center,
                                p_center,
                                cos,
                                dist,
                                iou,
                                color,
                                label_positions,
                                rng,
                                base_t=base_t,
                                cache_update_cb=_upd,
                            )
                            line_counter += 1

                # Write frame
                writer.write_frame(canvas)

                # Update prev state
                prev_frame_scaled = cur_scaled
                prev_dets = cur_dets
                prev_idx = frame_index
    return None
