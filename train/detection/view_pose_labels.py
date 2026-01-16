#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import yaml

from screen_utils import get_screen_size, overlay_screen_warning


SUPPORTED_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
KP_NAMES = ['boom_mast', 'mast_tip']
KP_COLORS_BGR = [
    (0, 165, 255),  # orange
    (255, 0, 255),  # magenta
]


@dataclass(frozen=True)
class YoloBBox:
    cls_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy_abs(self, *, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        cx = self.cx * img_w
        cy = self.cy * img_h
        bw = self.w * img_w
        bh = self.h * img_h
        x1 = int(round(cx - bw / 2.0))
        y1 = int(round(cy - bh / 2.0))
        x2 = int(round(cx + bw / 2.0))
        y2 = int(round(cy + bh / 2.0))
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(x1 + 1, min(img_w, x2))
        y2 = max(y1 + 1, min(img_h, y2))
        return x1, y1, x2, y2


@dataclass(frozen=True)
class Kp:
    x: float
    y: float
    v: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Viewer: render detection bboxes + 2 keypoints from a pose project on the original full-frame images.\n'
            'Keys: ,/. prev/next, Space next, s save render, Esc quit'
        )
    )
    p.add_argument('--src', type=Path, required=True, help='Detection dataset root (images + bbox labels).')
    p.add_argument('--pose', type=Path, required=True, help='Pose project dir (pose_index.yaml + labels_pose/...).')
    p.add_argument('--split', choices=['train', 'val', 'both'], default='val')
    p.add_argument('--max-side', type=int, default=1400, help='Max side for display resizing.')
    p.add_argument('--only-labeled', action='store_true', help='Only show samples that have pose labels.')
    p.add_argument(
        '--label-source',
        choices=['manual', 'pseudo', 'auto'],
        default='manual',
        help="Which labels to render: manual=labels_pose, pseudo=labels_pose_pseudo, auto=prefer manual else pseudo.",
    )
    p.add_argument('--out', type=Path, default=None, help='Optional output dir to save rendered images on demand (key: s).')
    return p.parse_args()


def _index_path(pose_dir: Path) -> Path:
    return pose_dir / 'pose_index.yaml'


def _load_index(pose_dir: Path) -> list[dict]:
    idx_path = _index_path(pose_dir)
    if not idx_path.exists():
        raise SystemExit(f'Missing pose index: {idx_path}')
    payload = yaml.safe_load(idx_path.read_text(encoding='utf-8')) or {}
    items = payload.get('items', [])
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, dict)]


def _pose_label_path(pose_dir: Path, *, split: str, key: str) -> Path:
    return pose_dir / 'labels_pose' / split / f'{key}.txt'

def _pseudo_label_path(pose_dir: Path, *, split: str, key: str) -> Path:
    return pose_dir / 'labels_pose_pseudo' / split / f'{key}.txt'


def _read_bboxes(label_path: Path) -> list[YoloBBox]:
    if not label_path.exists():
        return []
    out: list[YoloBBox] = []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            continue
        out.append(YoloBBox(cls_id=cls_id, cx=cx, cy=cy, w=w, h=h))
    return out


def _read_pose_lines(pose_path: Path) -> list[tuple[YoloBBox, list[Kp]]]:
    """
    Expected line format:
      cls cx cy w h kpx0 kpy0 v0 kpx1 kpy1 v1
    """
    if not pose_path.exists():
        return []
    out: list[tuple[YoloBBox, list[Kp]]] = []
    for line in pose_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 5 + 2 * 3:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            k0x, k0y, v0 = float(parts[5]), float(parts[6]), int(float(parts[7]))
            k1x, k1y, v1 = float(parts[8]), float(parts[9]), int(float(parts[10]))
        except ValueError:
            continue
        bbox = YoloBBox(cls_id=cls_id, cx=cx, cy=cy, w=w, h=h)
        kps = [Kp(x=k0x, y=k0y, v=1 if v0 > 0 else 0), Kp(x=k1x, y=k1y, v=1 if v1 > 0 else 0)]
        out.append((bbox, kps))
    return out


def _resize_to_max_side(img, max_side: int) -> tuple:
    h, w = img.shape[:2]
    max_side = max(1, int(max_side))
    scale = float(max_side) / float(max(h, w, 1))
    if scale >= 1.0:
        return img, 1.0
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    out = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return out, scale


def main() -> int:
    args = _parse_args()
    src_dir = Path(args.src)
    pose_dir = Path(args.pose)
    if not src_dir.exists():
        raise SystemExit(f'--src does not exist: {src_dir}')
    if not pose_dir.exists():
        raise SystemExit(f'--pose does not exist: {pose_dir}')

    splits = ['train', 'val'] if str(args.split) == 'both' else [str(args.split)]
    items = _load_index(pose_dir)
    items = [it for it in items if str(it.get('split', '')) in splits]
    if not items:
        raise SystemExit('No items found for chosen split(s).')

    if args.out is not None:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    def has_pose(it: dict) -> bool:
        key = str(it.get('key', ''))
        sp = str(it.get('split', 'train'))
        if not key:
            return False
        manual = _pose_label_path(pose_dir, split=sp, key=key)
        pseudo = _pseudo_label_path(pose_dir, split=sp, key=key)
        src = str(args.label_source)
        if src == 'manual':
            return manual.exists()
        if src == 'pseudo':
            return pseudo.exists()
        return manual.exists() or pseudo.exists()

    if bool(args.only_labeled):
        items = [it for it in items if has_pose(it)]
        if not items:
            raise SystemExit('No labeled samples found for chosen split(s).')

    idx = 0
    win = 'pose-viewer'
    cv2.namedWindow(win)
    screen_size = get_screen_size()

    while True:
        idx = max(0, min(idx, len(items) - 1))
        it = items[idx]
        key = str(it.get('key', ''))
        sp = str(it.get('split', 'train'))
        rel = str(it.get('src_rel', ''))
        img_path = src_dir / Path(rel)
        if not img_path.exists():
            idx = min(len(items) - 1, idx + 1)
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            idx = min(len(items) - 1, idx + 1)
            continue

        img_h, img_w = img.shape[:2]
        bbox_lines = _read_bboxes(img_path.with_suffix('.txt'))
        manual_path = _pose_label_path(pose_dir, split=sp, key=key)
        pseudo_path = _pseudo_label_path(pose_dir, split=sp, key=key)
        src = str(args.label_source)
        if src == 'manual':
            pose_path = manual_path
        elif src == 'pseudo':
            pose_path = pseudo_path
        else:
            pose_path = manual_path if manual_path.exists() else pseudo_path
        pose_lines = _read_pose_lines(pose_path)

        disp, scale = _resize_to_max_side(img, int(args.max_side))
        canvas = disp.copy()

        title = f'{sp.upper()} {idx+1}/{len(items)}  {img_path.name}  key={key}'
        cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Prefer pose labels as source of bbox+kp (it should match GT bboxes).
        lines_to_draw: list[tuple[YoloBBox, list[Kp]]]
        if pose_lines:
            lines_to_draw = pose_lines
        else:
            lines_to_draw = [(b, [Kp(0.0, 0.0, 0), Kp(0.0, 0.0, 0)]) for b in bbox_lines]

        for bi, (bbox, kps) in enumerate(lines_to_draw):
            x1, y1, x2, y2 = bbox.to_xyxy_abs(img_w=img_w, img_h=img_h)
            x1 = int(round(x1 * scale))
            y1 = int(round(y1 * scale))
            x2 = int(round(x2 * scale))
            y2 = int(round(y2 * scale))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                canvas,
                str(bi + 1),
                (x1 + 4, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            for ki, kp in enumerate(kps[:2]):
                if kp.v <= 0:
                    continue
                px = int(round(kp.x * img_w * scale))
                py = int(round(kp.y * img_h * scale))
                color = KP_COLORS_BGR[ki]
                cv2.circle(canvas, (px, py), 7, (0, 0, 0), -1)
                cv2.circle(canvas, (px, py), 5, color, -1)
                cv2.putText(
                    canvas,
                    str(ki + 1),
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Legend
        y = 56
        for i, name in enumerate(KP_NAMES):
            color = KP_COLORS_BGR[i]
            cv2.circle(canvas, (18, y - 6), 7, color, -1)
            cv2.putText(canvas, f'{i+1}: {name}', (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            y += 26

        if pose_lines and bbox_lines and len(pose_lines) != len(bbox_lines):
            cv2.putText(
                canvas,
                f'WARN: pose lines ({len(pose_lines)}) != bbox lines ({len(bbox_lines)})',
                (10, canvas.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        canvas = overlay_screen_warning(canvas, screen_size)
        cv2.imshow(win, canvas)
        keycode = cv2.waitKey(0)

        if keycode == 27:  # Esc
            break
        if keycode == ord(','):
            idx = max(0, idx - 1)
        elif keycode in (ord('.'), ord(' ')):
            idx = min(len(items) - 1, idx + 1)
        elif keycode == ord('s') and args.out is not None:
            out_path = Path(args.out) / f'{key}.jpg'
            cv2.imwrite(str(out_path), canvas)

    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
