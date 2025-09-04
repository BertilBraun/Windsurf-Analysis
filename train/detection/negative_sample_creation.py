#!/usr/bin/env python3
"""
negative_sample_creation.py – simple tool to extract negative samples

Mouse:
    LMB click inside box : select that box
    LMB drag             : draw a new box (crop region)

Keys:
    r        : delete selected box (or last if none selected)
    Space    : save selected crops for this image, then go to next image
    , / .    : previous / next image
    Esc      : quit

Notes:
    - Saves cropped regions as images and writes an accompanying EMPTY YOLO label file
      (negative sample) with the same basename.
    - Ensures each saved image is at least 640x640 pixels by expanding the rectangle
      to the minimum size within image bounds and padding if necessary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from screen_utils import get_screen_size, overlay_screen_warning


SUPPORTED_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


@dataclass
class BBox:
    # Coordinates in ORIGINAL image pixel space
    x1: float
    y1: float
    x2: float
    y2: float

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class NegativeSampleCreator:
    def __init__(self, images_dir: Path, output_dir: Path, min_side: int = 640) -> None:
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.min_side = max(1, int(min_side))

        self.image_paths: List[Path] = self._collect_images(images_dir)
        if not self.image_paths:
            raise SystemExit(f'No images found in {images_dir}')

        self.index: int = 0
        self.img: Optional[np.ndarray] = None

        self.boxes: List[BBox] = []
        self.selected_idx: Optional[int] = None

        # drawing state
        self.is_drawing: bool = False
        self.start_x: int = 0
        self.start_y: int = 0
        self.mouse_x: int = 0
        self.mouse_y: int = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

        cv2.namedWindow('negative-sample-creator')
        cv2.setMouseCallback('negative-sample-creator', self._mouse_cb)

    # ---------- IO helpers -------------------------------------------------
    def _collect_images(self, root: Path) -> List[Path]:
        return [p for p in sorted(root.rglob('*')) if p.suffix.lower() in SUPPORTED_IMG_EXTS]

    def _load_image(self) -> None:
        img_path = self.image_paths[self.index]
        self.img = cv2.imread(str(img_path))
        if self.img is None:
            raise SystemExit(f'Failed to read image: {img_path}')
        self.boxes = []
        self.selected_idx = None

    def _next_output_path(self, src_image: Path, idx: int) -> Tuple[Path, Path]:
        stem = src_image.stem
        out_img = self.output_dir / f'neg_{stem}_{idx:04d}.jpg'
        out_lbl = out_img.with_suffix('.txt')
        return out_img, out_lbl

    # ---------- crop helpers ----------------------------------------------
    def _expand_to_min_size(
        self, x1: float, y1: float, x2: float, y2: float, H: int, W: int
    ) -> Tuple[int, int, int, int]:
        min_w = self.min_side
        min_h = self.min_side

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        if w < min_w:
            w = float(min_w)
        if h < min_h:
            h = float(min_h)

        # propose expanded box centered at (cx, cy)
        nx1 = int(round(cx - w * 0.5))
        ny1 = int(round(cy - h * 0.5))
        nx2 = int(round(cx + w * 0.5))
        ny2 = int(round(cy + h * 0.5))

        # clamp to image bounds while preserving requested size as much as possible
        dx_left = max(0, -nx1)
        dy_top = max(0, -ny1)
        dx_right = max(0, nx2 - (W - 1))
        dy_bottom = max(0, ny2 - (H - 1))

        nx1 += dx_left
        nx2 -= dx_right
        ny1 += dy_top
        ny2 -= dy_bottom

        # Ensure valid min size after clamping
        nx2 = max(nx2, nx1 + 1)
        ny2 = max(ny2, ny1 + 1)
        return nx1, ny1, nx2, ny2

    def _ensure_min_dims(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        pad_h = max(0, self.min_side - h)
        pad_w = max(0, self.min_side - w)
        if pad_h == 0 and pad_w == 0:
            return crop
        # Pad evenly on both sides (as evenly as possible)
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return cv2.copyMakeBorder(crop, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)

    def _save_crops(self) -> None:
        assert self.img is not None
        H, W = self.img.shape[:2]
        img_path = self.image_paths[self.index]

        for i, b in enumerate(self.boxes, start=1):
            x1, y1, x2, y2 = b.as_int_tuple()
            x1, y1, x2, y2 = self._expand_to_min_size(x1, y1, x2, y2, H, W)
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W - 1))
            y2 = max(y1 + 1, min(y2, H - 1))

            crop = self.img[y1:y2, x1:x2].copy()
            crop = self._ensure_min_dims(crop)

            out_img_path, out_lbl_path = self._next_output_path(img_path, i)
            # Avoid overwrite by bumping index until file does not exist
            inc = 0
            while out_img_path.exists() or out_lbl_path.exists():
                inc += 1
                out_img_path, out_lbl_path = self._next_output_path(img_path, i + inc)

            ok = cv2.imwrite(str(out_img_path), crop)
            if not ok:
                continue
            # Create empty label file for negative sample
            with open(out_lbl_path, 'w'):
                pass

    # ---------- drawing / UI ----------------------------------------------
    def _draw(self) -> None:
        assert self.img is not None
        canvas = self.img.copy()
        for i, b in enumerate(self.boxes):
            color = (0, 255, 0)
            thickness = 2
            if self.selected_idx is not None and i == self.selected_idx:
                color = (0, 0, 255)
                thickness = 3
            x1, y1, x2, y2 = b.as_int_tuple()
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # preview rectangle while drawing
        if self.is_drawing:
            x1 = min(self.start_x, self.mouse_x)
            y1 = min(self.start_y, self.mouse_y)
            x2 = max(self.start_x, self.mouse_x)
            y2 = max(self.start_y, self.mouse_y)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Warn if the display image is larger than the screen (allow modest margins)
        canvas = overlay_screen_warning(canvas, get_screen_size())

        title = (
            f'negative-sample-creator [{self.index + 1}/{len(self.image_paths)}] '
            f'{self.image_paths[self.index].name}  |  Boxes: {len(self.boxes)}'
        )
        cv2.setWindowTitle('negative-sample-creator', title)
        cv2.imshow('negative-sample-creator', canvas)

    def _mouse_cb(self, event, x, y, flags, param) -> None:
        self.mouse_x, self.mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            # Select if clicked inside any existing box (prefer last on ties)
            clicked_idx = None
            for i in reversed(range(len(self.boxes))):
                if self.boxes[i].contains(x, y):
                    clicked_idx = i
                    break
            if clicked_idx is not None:
                self.selected_idx = clicked_idx
                self.is_drawing = False
            else:
                # start drawing a new box
                self.is_drawing = True
                self.start_x, self.start_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            pass

        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.is_drawing = False
            x1 = float(min(self.start_x, x))
            y1 = float(min(self.start_y, y))
            x2 = float(max(self.start_x, x))
            y2 = float(max(self.start_y, y))
            if x2 > x1 + 1 and y2 > y1 + 1:
                self.boxes.append(BBox(x1, y1, x2, y2))
                self.selected_idx = len(self.boxes) - 1

    def _delete_selected(self) -> None:
        if not self.boxes:
            return
        idx = self.selected_idx if self.selected_idx is not None else len(self.boxes) - 1
        del self.boxes[idx]
        if not self.boxes:
            self.selected_idx = None
        else:
            self.selected_idx = min(idx, len(self.boxes) - 1)

    # ---------- main loop --------------------------------------------------
    def run(self) -> None:
        self._load_image()
        while True:
            # Allow closing via window close (X) button
            try:
                if cv2.getWindowProperty('negative-sample-creator', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            self._draw()
            key = cv2.waitKey(20)

            if key == 27:  # Esc
                break

            elif key == ord('r'):
                self._delete_selected()

            elif key == ord(' '):
                self._save_crops()
                # advance to next image after saving
                if self.index < len(self.image_paths) - 1:
                    self.index += 1
                self._load_image()

            # previous / next image
            elif key == ord(','):
                if self.index > 0:
                    self.index -= 1
                    self._load_image()
            elif key == ord('.'):
                if self.index < len(self.image_paths) - 1:
                    self.index += 1
                    self._load_image()

        cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('images_dir', type=Path, help='Directory containing images to extract negatives from')
    p.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory to save negative crops and empty labels (default: <images_dir>/negatives)',
    )
    p.add_argument(
        '--min-side',
        type=int,
        default=640,
        help='Ensure each saved crop has at least this size for both width and height',
    )
    args = p.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else (args.images_dir / 'negatives')
    tool = NegativeSampleCreator(args.images_dir, output_dir, min_side=args.min_side)
    tool.run()


if __name__ == '__main__':
    main()
