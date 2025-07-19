from __future__ import annotations

"""Debug‑rendering helpers
===========================

This module introduces two high‑level classes that act as the backbone for the
new debug‑video renderer:

* **`DebugCanvas`** – Owns the *whole* output image and is responsible for any
  drawing that requires global knowledge, e.g. collision‑free label placement
  or connector lines that span multiple sub‑views.
* **`DebugView`** – A lightweight façade that represents a *rectangular region*
  of the canvas (for example “current frame”, “previous frame”, a single cell
  in a 4×4 grid, …).  It handles coordinate transforms and delegates the actual
  drawing to its parent `DebugCanvas`.

Importantly all coordinates are in *canvas* coordinates of the original frame.
They will be converted to canvas coordinates of the debug view late, just before drawing using primitives.

The public API is intentionally minimal yet composable so that callers can
re‑use existing helper functions almost unchanged while gaining the flexibility
of an arbitrarily complex layout.

Example
-------
```python
canvas_img = np.zeros((2 * h, w, 3), np.uint8)
canvas = DebugCanvas(canvas_img)

cur_view  = canvas.create_view(0,      0, w, h, scale_x=0.5, scale_y=0.5)
prev_view = canvas.create_view(0,      h, w, h, scale_x=0.5, scale_y=0.5)

cur_centres  = cur_view.annotate_detections(cur_dets, BOX_COLOR_CUR)
prev_centres = prev_view.annotate_detections(prev_dets, BOX_COLOR_PREV)

canvas.draw_line_with_metrics(cur_centres, prev_centres, cur_dets, prev_dets)
cv2.imwrite("preview.png", canvas_img)
```

The snippet above recreates the classic two‑row layout without any of the old
special‑case code paths.  Switching to a 4×4 grid is as easy as adding more
`create_view` calls.
"""

import os
import cv2
import math
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict
from stabilize import VidStabWithoutVideoCapture


from common_types import BoundingBox, Detection, FrameIndex, Point, TrackId, cosine_similarity, Track

BOX_COLOR_CUR = (255, 255, 255)  # white
BOX_COLOR_PREV = (170, 170, 170)  # light grey
BOX_COLOR_PREV_TRANSFORMED = (120, 120, 170)  # grayish blue
ALPHA_OVERLAY = 0.5  # translucency for stabilised preview
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _clamp(value: int, min_v: int, max_v: int) -> int:
    """Clamp *value* between *min_v* and *max_v*."""
    return max(min_v, min(max_v, value))


def _generate_palette(n: int = 30, seed: int = 0) -> List[Tuple[int, int, int]]:
    """Return *n* bright BGR colours with deterministic shuffling."""
    colours: List[Tuple[int, int, int]] = []
    for i in range(n):
        h = int((i / n) * 179)  # OpenCV hue range [0, 179]
        s, v = 200, 255
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        colours.append(tuple(int(c) for c in bgr))
    rng = np.random.default_rng(seed)
    rng.shuffle(colours)
    return colours


class DebugCanvas:
    """Owns the global image and collision‑aware drawing primitives.

    Draw calls are in debug_view coordinates."""

    def __init__(self, canvas: np.ndarray, feed_w, feed_h, *, seed: int | None = None):
        self.canvas = canvas  # uint8 H×W×3 array
        self.h, self.w = canvas.shape[:2]
        # size of the original video feed
        self.feed_w, self.feed_h = feed_w, feed_h
        # placed labels in debug view coordinates --> used to avoid collision.
        self.label_positions: list[BoundingBox] = []  # placed labels (global)

        # Colour palette used by connector lines
        self._seed = int(seed) if seed is not None else 0
        self.palette: List[Tuple[int, int, int]] = _generate_palette(30, self._seed)
        self._rng = np.random.default_rng(self._seed ^ 0xA5A5A5A5)

    def create_view(self, x: int, y: int, w: int, h: int) -> DebugView:
        """Return a new view covering *rect* (x, y, w, h) in canvas coords."""
        return DebugView(self, x, y, w, h)

    # ---------------------------------------------------------------------
    # Global drawing ops ---------------------------------------------------

    def draw_text(
        self,
        text: str,
        origin_bl: Point,
        color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.4,
        thickness: int = 1,
        bg: bool = True,
    ) -> BoundingBox:
        (tw, th), _ = cv2.getTextSize(text, _FONT, font_scale, thickness)
        x, y = origin_bl
        text_bb = BoundingBox(x, y - th, x + tw, y)
        if bg:
            cv2.rectangle(self.canvas, (text_bb.x1, text_bb.y1), (text_bb.x2, text_bb.y2), (40, 40, 40), -1)
        cv2.putText(self.canvas, text, (x, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)
        return text_bb

    def draw_label(
        self,
        text: str,
        origin_bl: Point,
        color: Tuple[int, int, int],
        font_scale: float = 0.4,
        thickness: int = 1,
        bg: bool = True,
    ):
        """Draws text and reserves space to avoid overwriting."""
        bbox_text = self.draw_text(
            text,
            origin_bl,
            color,
            font_scale=font_scale,
            thickness=thickness,
            bg=bg,
        )
        self.label_positions.append(bbox_text)

    def draw_box(
        self,
        bbox: BoundingBox,
        color: Tuple[int, int, int],
        label: Optional[str] = None,
    ):
        cv2.rectangle(
            self.canvas,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            color,
            1,
            cv2.LINE_AA,
        )
        if label:
            self.draw_label(label, Point(bbox.x1, max(bbox.y1 - 2, 0)), color)

    def _choose_label_org_for_line_cached(
        self,
        text: str,
        p1: Point,
        p2: Point,
        font_scale: float = 0.35,
        thickness: int = 1,
        step: float = 0.1,
    ) -> Point:
        base_t = 0.25
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
            cx, cy = p1.interpolate(p2, t)
            org, bb_text = self._clamp_label_origin(text, cx, cy, font_scale, thickness)
            if not any(bb.overlaps(bb_text) for bb in self.label_positions):
                return org
        # fallback random
        t = float(self._rng.random())
        cx, cy = p1.interpolate(p2, t)
        org, _ = self._clamp_label_origin(text, cx, cy, font_scale, thickness)
        return org

    def _clamp_label_origin(
        self,
        text: str,
        x: int,
        y: int,
        font_scale: float,
        thickness: int,
    ) -> tuple[Point, BoundingBox]:
        """Clamp label origin to canvas bounds and return the text bounding box."""
        assert x >= 0 and y >= 0
        (tw, th), _ = cv2.getTextSize(text, _FONT, font_scale, thickness)
        top = max(0, y - th)
        right = min(self.w, x + tw)
        left = right - tw
        bottom = min(self.h, y)
        assert left >= 0 and bottom >= 0
        org = Point(left, bottom)
        bbox = BoundingBox(left, top, right, bottom)
        return org, bbox

    def draw_line_with_label(self, start: Point, end: Point, text: str, color: Tuple[int, int, int]) -> None:
        """Draw a line between *start* and *end* with a label at the midpoint."""
        cv2.line(self.canvas, (start.x, start.y), (end.x, end.y), color, 1, cv2.LINE_AA)
        org = self._choose_label_org_for_line_cached(text, start, end)
        self.draw_label(text, org, color)


class DebugView:
    """Represents a rectangular sub‑region of a `DebugCanvas`. It is assumed that this region is associated with one display of the original video feed.

    Draw calls are made in detection coordinates. I.e. coordinates of the original video feed."""

    def __init__(
        self,
        canvas: DebugCanvas,
        # coordinates in DebugCanvas
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> None:
        self._c = canvas
        self.x, self.y, self.w, self.h = map(int, (x, y, w, h))

    def _feed_to_global(self, feed_x: int, feed_y: int) -> Point:
        """Feed‑space → debug-canvas‑space."""
        x_f = feed_x * self.w / self._c.feed_w + self.x
        y_f = feed_y * self.h / self._c.feed_h + self.y
        return Point(_clamp(int(round(x_f)), 0, self._c.w), _clamp(int(round(y_f)), 0, self._c.h))

    def _feed_bb_to_global(self, bbox: BoundingBox) -> BoundingBox:
        """Feed‑space BoundingBox → debug-canvas‑space BoundingBox."""
        x1, y1 = self._feed_to_global(bbox.x1, bbox.y1)
        x2, y2 = self._feed_to_global(bbox.x2, bbox.y2)
        return BoundingBox(x1, y1, x2, y2)

    def draw_box(
        self,
        bbox: BoundingBox,
        color: Tuple[int, int, int],
        label: Optional[str] = None,
    ):
        bb_transformed = self._feed_bb_to_global(bbox)
        self._c.draw_box(bb_transformed, color, label)

    def draw_label(
        self,
        text: str,
        origin_tl: Point,
        color: Tuple[int, int, int],
        font_scale: float = 0.4,
        thickness: int = 1,
        bg: bool = True,
    ):
        """Draws text and reserves space to avoid overwriting."""
        org = self._feed_to_global(*origin_tl)
        self._c.draw_label(text, org, color, font_scale=font_scale, thickness=thickness, bg=bg)

    def draw_text(
        self,
        text: str,
        origin_tl: Point,
        color: Tuple[int, int, int],
        font_scale: float = 0.4,
        thickness: int = 1,
        bg: bool = True,
    ):
        org = self._feed_to_global(*origin_tl)
        self._c.draw_text(text, org, color, font_scale=font_scale, thickness=thickness, bg=bg)


def generate_debug_video_worker_function(
    args: Tuple[
        List[Detection],  # all detections (flat list)
        List[Track],
        VidStabWithoutVideoCapture | None,  # stabilizer instance
        os.PathLike,  # input video path
        os.PathLike | str,  # output directory
    ],
) -> None:
    """Multiprocessing worker that writes the debug video to disk using the DebugCanvas API."""

    detections, tracks, stabilizer_in, input_path, output_dir = args
    if stabilizer_in is not None:
        stabilizer = stabilizer_in.get_vid_stab(input_path)
        transforms = stabilizer.transforms
    else:
        transforms = None
    det_map = defaultdict(list)
    for det in detections:
        det_map[det.frame_idx].append(det)

    track_detections_per_frame: dict[FrameIndex, list[Tuple[TrackId, Detection]]] = defaultdict(list)
    for track in tracks:
        for d in track.sorted_detections:
            track_detections_per_frame[d.frame_idx].append((track.track_id, d))

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{input_path.stem}+00_debug.mp4'
    logging.info(f'Generating debug video for {input_path} with {len(detections)} detections. Writing to {out_path}')

    from video_io import VideoReader, VideoWriter  # type: ignore

    seed = abs(hash(str(input_path)))

    with VideoReader(input_path) as reader:
        props = reader.get_properties()  # width, height, fps, total_frames
        feed_w, feed_h = props.width, props.height
        sw, sh = max(1, feed_w // 2), max(1, feed_h // 2)
        dbg_w, dbg_h = sw, sh * 2

        with VideoWriter(out_path, dbg_w, dbg_h, props.fps) as writer:
            prev_scaled: Optional[np.ndarray] = None
            prev_dets: List[Tuple[TrackId, Detection]] = []

            for f_idx, frame in tqdm(reader.read_frames(), total=props.total_frames, desc='Debug render'):
                cur_scaled = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)
                bottom = prev_scaled if prev_scaled is not None else np.zeros_like(cur_scaled)

                # Create canvas and views
                canvas_img = np.zeros((dbg_h, dbg_w, 3), dtype=cur_scaled.dtype)
                canvas_img[0:sh] = cur_scaled
                canvas_img[sh : sh * 2] = bottom

                canvas = DebugCanvas(canvas_img, feed_w, feed_h, seed=seed)
                cur_view = canvas.create_view(0, 0, sw, sh)
                prev_view = canvas.create_view(0, sh, sw, sh)

                # Frame index labels
                canvas.draw_text(f't={f_idx}', Point(5, 15), color=(0, 255, 0), bg=True)
                if prev_scaled is not None:
                    canvas.draw_text(f't={f_idx - 1}', Point(5, sh + 15), color=(0, 255, 0), bg=True)

                # Draw detections
                cur_dets = det_map.get(f_idx, [])
                for d in cur_dets:
                    cur_view.draw_box(d.bbox, BOX_COLOR_CUR, label=f'{d.confidence:.2f}')
                prev_dets = track_detections_per_frame.get(f_idx - 1, [])
                if prev_dets is not None:
                    for t_id, d in prev_dets:
                        prev_view.draw_box(d.bbox, BOX_COLOR_CUR, label=f'{t_id}')
                        cur_view.draw_box(d.bbox, BOX_COLOR_PREV)

                for det_c in cur_dets:
                    cur_c_global = cur_view._feed_to_global(*det_c.bbox.center)
                    for _t_id, det_p in prev_dets:
                        prev_c_global = prev_view._feed_to_global(*det_p.bbox.center)
                        cos = (
                            cosine_similarity(det_c.feat, det_p.feat)
                            if det_c.feat is not None and det_p.feat is not None
                            else float('nan')
                        )
                        dist = math.hypot(
                            det_c.bbox.center.x - det_p.bbox.center.x,
                            det_c.bbox.center.y - det_p.bbox.center.y,
                        )
                        iou = det_c.bbox.iou(det_p.bbox)
                        txt = (
                            f'c={cos if cos is not None and not math.isnan(cos) else "n/a":.2f} '
                            f'iou={iou if iou is not None and not math.isnan(iou) else "n/a":.2f} '
                            f'd={dist if dist is not None and not math.isnan(dist) else "n/a":.0f}'
                        )
                        canvas.draw_line_with_label(
                            cur_c_global,
                            prev_c_global,
                            text=txt,
                            color=BOX_COLOR_CUR,
                        )

                writer.write_frame(canvas_img)
                prev_scaled = cur_scaled

    return None
