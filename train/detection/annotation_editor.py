#!/usr/bin/env python3
"""
annotator_images.py – simple image dataset bbox editor

Mouse:
    LMB click inside box : select that box
    LMB drag             : draw a new box

Keys:
    r        : delete selected box (or last if none selected)
    Space    : save labels (.txt) for this image
    , / .    : previous / next image
    w/a/s/d  : move/resize selected box edge (mode: grow)
    W/A/S/D  : move/resize selected box edge (mode: shrink)
    Esc      : quit
    backspace: delete sample

Notes:
    - Loads and saves YOLO format labels: class cx cy w h (normalized)
    - Works on a resized display but preserves original image dimensions in saved labels
"""

from __future__ import annotations

import argparse
import os
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from screen_utils import get_screen_size, overlay_screen_warning


SUPPORTED_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


@dataclass
class BBox:
    # Coordinates in DISPLAY space (after scaling). We'll convert on save.
    x1: float
    y1: float
    x2: float
    y2: float
    cls_id: int = 0

    def as_int_tuple(self) -> Tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class ImageBboxEditor:
    def __init__(self, images_dir: Path, max_side: int = 2048) -> None:
        self.images_dir = images_dir
        self.max_side = max_side

        self.image_paths: List[Path] = self._collect_images(images_dir)
        if not self.image_paths:
            raise SystemExit(f'No images found in {images_dir}')

        self.index: int = 0
        self.img: Optional[np.ndarray] = None  # image

        self.boxes: List[BBox] = []
        self.selected_idx: Optional[int] = None

        # drawing state
        self.is_drawing: bool = False
        self.start_x: int = 0
        self.start_y: int = 0
        self.mouse_x: int = 0
        self.mouse_y: int = 0

        cv2.namedWindow('bbox-editor')
        cv2.setMouseCallback('bbox-editor', self._mouse_cb)

    # ---------- IO helpers -------------------------------------------------
    def _collect_images(self, root: Path) -> List[Path]:
        imgs = [p for p in sorted(root.rglob('*')) if p.suffix.lower() in SUPPORTED_IMG_EXTS]
        return imgs

    def _label_path_for(self, image_path: Path) -> Path:
        return image_path.with_suffix('.txt')

    def _load_image_and_labels(self) -> None:
        img_path = self.image_paths[self.index]
        self.img = cv2.imread(str(img_path))
        if self.img is None:
            raise SystemExit(f'Failed to read image: {img_path}')

        self.boxes = self._load_labels_for(img_path)
        self.selected_idx = len(self.boxes) - 1 if self.boxes else None

    def _load_labels_for(self, image_path: Path) -> List[BBox]:
        label_path = self._label_path_for(image_path)
        boxes: List[BBox] = []
        if not label_path.exists():
            return boxes

        assert self.img is not None
        H_orig, W_orig = self.img.shape[:2]
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                # Convert normalized to absolute ORIGINAL pixels
                abs_w = bw * W_orig
                abs_h = bh * H_orig
                center_x = cx * W_orig
                center_y = cy * H_orig
                x1 = center_x - abs_w / 2.0
                y1 = center_y - abs_h / 2.0
                x2 = center_x + abs_w / 2.0
                y2 = center_y + abs_h / 2.0
                boxes.append(BBox(x1, y1, x2, y2, cls_id))
        return boxes

    def _save_labels(self) -> None:
        img_path = self.image_paths[self.index]
        label_path = self._label_path_for(img_path)
        assert self.img is not None
        H_orig, W_orig = self.img.shape[:2]

        with open(label_path, 'w') as f:
            for b in self.boxes:
                x1 = b.x1
                y1 = b.y1
                x2 = b.x2
                y2 = b.y2
                # Clamp
                x1 = max(0.0, min(x1, W_orig - 1.0))
                y1 = max(0.0, min(y1, H_orig - 1.0))
                x2 = max(x1 + 1.0, min(x2, W_orig - 1.0))
                y2 = max(y1 + 1.0, min(y2, H_orig - 1.0))

                bw = x2 - x1
                bh = y2 - y1
                cx = x1 + bw / 2.0
                cy = y1 + bh / 2.0

                f.write(f'{b.cls_id} {cx / W_orig:.6f} {cy / H_orig:.6f} {bw / W_orig:.6f} {bh / H_orig:.6f}\n')

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

        title = f'bbox-editor [{self.index + 1}/{len(self.image_paths)}] {self.image_paths[self.index].name}'
        cv2.setWindowTitle('bbox-editor', title)
        cv2.imshow('bbox-editor', canvas)

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
            pass  # just update preview via _draw()

        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing:
            self.is_drawing = False
            x1 = float(min(self.start_x, x))
            y1 = float(min(self.start_y, y))
            x2 = float(max(self.start_x, x))
            y2 = float(max(self.start_y, y))
            # enforce minimum size
            if x2 > x1 + 1 and y2 > y1 + 1:
                self.boxes.append(BBox(x1, y1, x2, y2, 0))
                self.selected_idx = len(self.boxes) - 1

    def _adjust_selected(self, dx1=0, dy1=0, dx2=0, dy2=0) -> None:
        if not self.boxes:
            return
        idx = self.selected_idx if self.selected_idx is not None else len(self.boxes) - 1
        self.selected_idx = idx
        b = self.boxes[idx]
        assert self.img is not None
        H, W = self.img.shape[:2]

        # step based on bbox size
        box_w = max(1.0, abs(b.x2 - b.x1))
        box_h = max(1.0, abs(b.y2 - b.y1))
        step_x = max(1.0, 0.01 * box_w)
        step_y = max(1.0, 0.01 * box_h)

        nx1 = max(0.0, min(b.x1 + dx1 * step_x, b.x2 - 1.0))
        ny1 = max(0.0, min(b.y1 + dy1 * step_y, b.y2 - 1.0))
        nx2 = min(W - 1.0, max(b.x2 + dx2 * step_x, nx1 + 1.0))
        ny2 = min(H - 1.0, max(b.y2 + dy2 * step_y, ny1 + 1.0))
        self.boxes[idx] = BBox(nx1, ny1, nx2, ny2, b.cls_id)

    def _delete_selected(self) -> None:
        if not self.boxes:
            return
        idx = self.selected_idx if self.selected_idx is not None else len(self.boxes) - 1
        del self.boxes[idx]
        if not self.boxes:
            self.selected_idx = None
        else:
            self.selected_idx = min(idx, len(self.boxes) - 1)

    def _delete_image(self) -> None:
        os.remove(self.image_paths[self.index])
        os.remove(self._label_path_for(self.image_paths[self.index]))
        self.image_paths.pop(self.index)
        if self.index == len(self.image_paths):
            self.index -= 1
        self._load_image_and_labels()

    # ---------- main loop --------------------------------------------------
    def run(self) -> None:
        self._load_image_and_labels()
        while True:
            # Allow closing via window close (X) button
            try:
                if cv2.getWindowProperty('bbox-editor', cv2.WND_PROP_VISIBLE) < 1:
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
                # Save labels, then advance to next image
                self._save_labels()
                if self.index < len(self.image_paths) - 1:
                    self.index += 1
                self._load_image_and_labels()

            # adjust selected
            elif (key) == ord('w'):
                self._adjust_selected(dy1=-1)
            elif key == ord('W'):
                self._adjust_selected(dy1=1)
            elif key == ord('a'):
                self._adjust_selected(dx1=-1)
            elif key == ord('A'):
                self._adjust_selected(dx1=1)
            elif key == ord('s'):
                self._adjust_selected(dy2=1)
            elif key == ord('S'):
                self._adjust_selected(dy2=-1)
            elif key == ord('d'):
                self._adjust_selected(dx2=1)
            elif key == ord('D'):
                self._adjust_selected(dx2=-1)

            elif key == ord('n'):
                # select next box
                self.selected_idx = min(self.selected_idx or 0 + 1, len(self.boxes) - 1)
            elif key == ord('p'):
                # select previous box
                self.selected_idx = max(self.selected_idx or 0 - 1, 0)

            # previous / next image
            elif key == ord(','):
                if self.index > 0:
                    self.index -= 1
                    self._load_image_and_labels()
            elif key == ord('.'):
                if self.index < len(self.image_paths) - 1:
                    self.index += 1
                    self._load_image_and_labels()

            elif key == 8:
                self._delete_image()

        cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('images_dir', type=Path, help='Directory containing images and YOLO .txt labels')
    p.add_argument(
        '--max-side', type=int, default=2048, help='Resize longest side for display (labels saved in original size)'
    )
    args = p.parse_args()
    editor = ImageBboxEditor(args.images_dir, max_side=args.max_side)
    editor.run()


if __name__ == '__main__':
    main()
