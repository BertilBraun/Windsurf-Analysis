"""Refactored debug‐video renderer
=================================

Key improvements
----------------
1. **Modular frame preparation & annotation** – Current‑ and previous‑frame drawing logic
   is isolated in helper functions so it can be unit‑tested or reused in other
   pipelines.
2. **Optional stabilised overlay** – Previous‑frame detections can be projected into
   the current frame using VidStab motion estimates.  The overlay is drawn in a
   translucent grey (α = 0.5) so that users can visually judge the effect of
   stabilisation.
3. **Slimmer main loop** – The worker now focuses on I/O and orchestration;
   all drawing happens in helpers that accept simple, declarative arguments.
4. **Backwards compatible** – If no `transforms` array is supplied the overlay is
   simply skipped and the behaviour matches the original implementation.

To integrate, either import the helper functions you need or call the
`generate_debug_video_worker_function` directly (its signature has one extra
parameter – see below).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Sequence
from collections import defaultdict
import math
import os
import logging

import cv2
import numpy as np
from tqdm import tqdm
from vidstab import VidStab

from common_types import Detection, BoundingBox, cosine_similarity
from helpers import log_and_reraise

# ---------------------------------------------------------------------------
# Basic configuration --------------------------------------------------------

BOX_COLOR_CUR = (255, 255, 255)      # white
BOX_COLOR_PREV = (170, 170, 170)     # light grey
aLPHA_OVERLAY = 0.5                  # translucency for stabilised preview
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------------------------------------------------------
# Colour utilities -----------------------------------------------------------


def _generate_palette(n: int = 30, seed: int = 0) -> List[Tuple[int, int, int]]:
    """Return *n* bright BGR colours with deterministic shuffling."""
    colours: List[Tuple[int, int, int]] = []
    for i in range(n):
        h = int((i / n) * 179)          # OpenCV hue range [0, 179]
        s, v = 200, 255
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        colours.append(tuple(int(c) for c in bgr))
    rng = np.random.default_rng(seed)
    rng.shuffle(colours)
    return colours

# ---------------------------------------------------------------------------
# Text & bounding‑box helpers ------------------------------------------------


def _measure_text(text: str, font_scale: float = 0.4, thickness: int = 1) -> Tuple[int, int, int]:
    (size, base) = cv2.getTextSize(text, _FONT, font_scale, thickness)
    w, h = size
    return w, h, base


def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], margin: int = 2) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1 -= margin; ay1 -= margin; ax2 += margin; ay2 += margin
    bx1 -= margin; by1 -= margin; bx2 += margin; by2 += margin
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
    """Draw *text* on *img* and return its rectangle (x1, y1, x2, y2)."""
    tw, th, base = _measure_text(text, font_scale, thickness)
    x, y = org
    tl, br = (x, y - th - base), (x + tw, y + base)
    if bg:
        cv2.rectangle(img, tl, br, (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)
    return tl[0], tl[1], br[0], br[1]


# ---------------------------------------------------------------------------
# Frame‑level helpers --------------------------------------------------------


def _scale_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Return *frame* resized isotropically by *scale*."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _draw_box(
    img: np.ndarray,
    box: BoundingBox,
    color: Tuple[int, int, int],
    label_positions: List[Tuple[int, int, int, int]],
    *,
    label: Optional[str] = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    y_offset: int = 0,
) -> Tuple[int, int]:
    """Draw a bounding box and optional label; return its centre in *img* coords."""
    x1 = int(box.x1 * scale_x)
    y1 = int(box.y1 * scale_y) + y_offset
    x2 = int(box.x2 * scale_x)
    y2 = int(box.y2 * scale_y) + y_offset
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    if label:
        rect = _draw_text(img, label, (x1, max(y1 - 2, 0)), color)
        label_positions.append(rect)
    return x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2


def _annotate_detections(
    canvas: np.ndarray,
    detections: Sequence[Detection],
    box_color: Tuple[int, int, int],
    label_positions: List[Tuple[int, int, int, int]],
    *,
    scale_x: float,
    scale_y: float,
    y_offset: int,
) -> List[Tuple[int, int]]:
    """Draw *detections* on *canvas*; return list of box centres."""
    centres: List[Tuple[int, int]] = []
    for det in detections:
        centres.append(
            _draw_box(
                canvas,
                det.bbox,
                box_color,
                label_positions,
                scale_x=scale_x,
                scale_y=scale_y,
                y_offset=y_offset,
            )
        )
    return centres


# ---------------------------------------------------------------------------
# Connector‑line helpers (unchanged logic, now packaged) ---------------------


def _point_along_line(p1: Tuple[int, int], p2: Tuple[int, int], t: float) -> Tuple[int, int]:
    return int(round(p1[0] + t * (p2[0] - p1[0]))), int(round(p1[1] + t * (p2[1] - p1[1])))


def _clamp_label_org(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float,
    thickness: int,
):
    h, w = img.shape[:2]
    tw, th, base = _measure_text(text, font_scale, thickness)
    ox = int(round(x - tw / 2))
    oy = int(round(y + th / 2))
    ox = max(0, min(ox, w - tw))
    oy = max(th + base, min(oy, h - 1))
    return (ox, oy - th - base, ox + tw, oy + base, (ox, oy))


def _choose_label_org_for_line_cached(
    img: np.ndarray,
    text: str,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    existing: List[Tuple[int, int, int, int]],
    base_t: float,
    rng: np.random.Generator,
    *,
    font_scale: float = 0.35,
    thickness: int = 1,
    step: float = 0.1,
) -> Tuple[Tuple[int, int], float]:
    """Collision‑aware label placement with cached preferred *t*."""
    cands: List[float] = [t for t in (base_t,) if 0.0 <= t <= 1.0]
    k, up_done, dn_done = 1, False, False
    while not (up_done and dn_done):
        tu, td = base_t + step * k, base_t - step * k
        if tu > 1.0:
            up_done = True
        else:
            cands.append(tu)
        if td < 0.0:
            dn_done = True
        else:
            cands.append(td)
        k += 1
        if k > 20:
            break
    for t in cands:
        cx, cy = _point_along_line(p1, p2, t)
        rx1, ry1, rx2, ry2, org = _clamp_label_org(img, text, cx, cy, font_scale, thickness)
        if not any(_rects_overlap((rx1, ry1, rx2, ry2), e) for e in existing):
            return org, t
    # fallback random
    t = float(rng.random())
    cx, cy = _point_along_line(p1, p2, t)
    _, _, _, _, org = _clamp_label_org(img, text, cx, cy, font_scale, thickness)
    return org, t


def _draw_line_with_metrics(
    img: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    cos: float | None,
    dist: float | None,
    iou: float | None,
    color: Tuple[int, int, int],
    label_positions: List[Tuple[int, int, int, int]],
    rng: np.random.Generator,
    *,
    base_t: float,
    cache_update_cb,
):
    cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
    txt = f"c={cos if cos is not None and not math.isnan(cos) else 'n/a':.2f} " \
          f"iou={iou if iou is not None and not math.isnan(iou) else 'n/a':.2f} " \
          f"d={dist if dist is not None and not math.isnan(dist) else 'n/a':.0f}"
    org, new_t = _choose_label_org_for_line_cached(img, txt, p1, p2, label_positions, base_t, rng)
    rect = _draw_text(img, txt, org, color, font_scale=0.35)
    label_positions.append(rect)
    cache_update_cb(new_t)


# ---------------------------------------------------------------------------
# Stabilised overlay helper --------------------------------------------------


def _draw_stabilised_prev_boxes(
    canvas: np.ndarray,
    prev_dets: Sequence[Detection],
    transform: Sequence[float] | None,
    label_positions: List[Tuple[int, int, int, int]],
    *,
    scale_x: float,
    scale_y: float,
    y_offset: int = 0,
) -> None:
    """Project *prev_dets* into current‑frame coordinates and draw translucently.

    Only translation (dx, dy) is applied; rotation (da) is ignored for simplicity.
    Set `transform=None` to skip overlay (e.g. at frame 0).
    """
    if transform is None:
        return
    dx, dy = transform[0], transform[1]
    overlay = canvas.copy()
    for det in prev_dets:
        box = det.bbox
        shifted = BoundingBox(
            x1=box.x1 + dx,
            y1=box.y1 + dy,
            x2=box.x2 + dx,
            y2=box.y2 + dy,
        )
        _draw_box(
            overlay,
            shifted,
            BOX_COLOR_PREV,
            label_positions,
            scale_x=scale_x,
            scale_y=scale_y,
            y_offset=y_offset,
        )
    # Blend overlay into canvas
    cv2.addWeighted(overlay, aLPHA_OVERLAY, canvas, 1 - aLPHA_OVERLAY, 0, dst=canvas)


# ---------------------------------------------------------------------------
# Main worker ----------------------------------------------------------------


@log_and_reraise
def generate_debug_video_worker_function(
    args: tuple[
        List[Detection],         # all detections (flat list)           ┐
        VidStab | None,                # stabiliser instance                   │ legacy order preserved
        os.PathLike,            # input video path                      │
        os.PathLike             # output directory                      ┘
    ]
) -> None:
    """Multiprocessing worker that writes the debug video to disk."""

    detections, stabiliser, input_path, output_dir = args
    transforms = stabiliser.transforms if stabiliser is not None else None
    logging.info(f"Generating debug video for {input_path} with {len(detections)} detections")
    det_map = defaultdict(list)
    for det in detections:
        det_map[det.frame_idx].append(det)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{input_path.stem}+00_debug.mp4"

    # Project‑specific reader/writer
    from video_io import VideoReader, VideoWriter  # type: ignore

    seed = abs(hash(str(input_path)))
    palette = _generate_palette(30, seed)
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    label_t_cache = [0.25] * len(palette)

    with VideoReader(input_path) as reader:
        props = reader.get_properties()  # width, height, fps, total_frames
        sw, sh = max(1, props.width // 2), max(1, props.height // 2)
        dbg_w, dbg_h = sw, sh * 2

        with VideoWriter(out_path, dbg_w, dbg_h, props.fps) as writer:
            prev_scaled: Optional[np.ndarray] = None
            prev_dets: List[Detection] = []
            line_counter = 0

            for f_idx, frame in tqdm(reader.read_frames(), total=props.total_frames, desc="Debug render"):
                label_positions: List[Tuple[int, int, int, int]] = []

                cur_scaled = _scale_frame(frame)
                bottom = prev_scaled if prev_scaled is not None else np.zeros_like(cur_scaled)

                canvas = np.zeros((dbg_h, dbg_w, 3), dtype=cur_scaled.dtype)
                canvas[0:sh] = cur_scaled
                canvas[sh:sh * 2] = bottom

                # Frame index labels
                label_positions.append(_draw_text(canvas, f"t={f_idx}", (5, 15), (0, 255, 0), bg=True))
                if prev_scaled is not None:
                    label_positions.append(_draw_text(canvas, f"t={f_idx-1}", (5, sh + 15), (0, 255, 0), bg=True))

                # Draw detections
                cur_dets = det_map.get(f_idx, [])
                cur_centres = _annotate_detections(canvas, cur_dets, BOX_COLOR_CUR, label_positions,
                                                   scale_x=0.5, scale_y=0.5, y_offset=0)
                prev_centres = _annotate_detections(canvas, prev_dets, BOX_COLOR_PREV, label_positions,
                                                    scale_x=0.5, scale_y=0.5, y_offset=sh)

                # Optional stabilised overlay (prev → current)
                transform = None
                if transforms is not None and f_idx > 0 and len(transforms) >= f_idx:
                    transform = transforms[f_idx - 1]
                _draw_stabilised_prev_boxes(canvas, prev_dets, transform, label_positions,
                                            scale_x=0.5, scale_y=0.5, y_offset=0)

                # Connector lines with metrics
                if cur_dets and prev_dets:
                    for i, det_c in enumerate(cur_dets):
                        for j, det_p in enumerate(prev_dets):
                            cos = cosine_similarity(det_c.feat, det_p.feat) if det_c.feat is not None and det_p.feat is not None else float('nan')
                            dist = math.hypot(det_c.bbox.center.x - det_p.bbox.center.x,
                                              det_c.bbox.center.y - det_p.bbox.center.y)
                            iou = det_c.bbox.iou(det_p.bbox)

                            colour_idx = line_counter % len(palette)
                            colour = palette[colour_idx]
                            base_t = label_t_cache[colour_idx]

                            _draw_line_with_metrics(
                                canvas,
                                cur_centres[i],
                                prev_centres[j],
                                cos,
                                dist,
                                iou,
                                colour,
                                label_positions,
                                rng,
                                base_t=base_t,
                                cache_update_cb=lambda new_t, idx=colour_idx: label_t_cache.__setitem__(idx, new_t),
                            )
                            line_counter += 1

                writer.write_frame(canvas)

                # Update prev‑state
                prev_scaled, prev_dets = cur_scaled, cur_dets

    return None
