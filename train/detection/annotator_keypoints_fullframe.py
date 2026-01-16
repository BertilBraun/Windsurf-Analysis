#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import yaml

from screen_utils import get_screen_size, overlay_screen_warning


SUPPORTED_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DISPLAY_MAX_SIDE = 900
MIN_BBOX_SIDE_PX = 5.0
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

    def to_xyxy_abs(self, *, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        cx = self.cx * img_w
        cy = self.cy * img_h
        bw = self.w * img_w
        bh = self.h * img_h
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        return x1, y1, x2, y2


@dataclass
class Kp:
    x: float = 0.0  # normalized
    y: float = 0.0  # normalized
    v: int = 0  # 0/1


def _slug_from_relpath(rel: Path) -> str:
    raw = rel.as_posix()
    raw = raw.rsplit('.', 1)[0]
    raw = raw.replace('/', '__')
    out = []
    for ch in raw:
        if ch.isalnum() or ch in '._-':
            out.append(ch)
        else:
            out.append('_')
    s = ''.join(out).strip('_')
    return s or 'sample'


def _collect_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_IMG_EXTS:
            continue
        out.append(p)
    return out


def _read_bboxes(label_path: Path) -> list[YoloBBox]:
    if not label_path.exists():
        return []
    bboxes: list[YoloBBox] = []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            continue
        bboxes.append(YoloBBox(cls_id=cls_id, cx=cx, cy=cy, w=w, h=h))
    return bboxes


def _bbox_xyxy_disp(b: YoloBBox, *, img_w: int, img_h: int, scale: float) -> tuple[int, int, int, int]:
    cx = b.cx * img_w
    cy = b.cy * img_h
    bw = b.w * img_w
    bh = b.h * img_h
    x1 = (cx - bw / 2.0) * scale
    y1 = (cy - bh / 2.0) * scale
    x2 = (cx + bw / 2.0) * scale
    y2 = (cy + bh / 2.0) * scale
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _resize_to_max_side(img, max_side: int) -> tuple:
    h, w = img.shape[:2]
    max_side = max(1, int(max_side))
    scale = float(max_side) / float(max(h, w, 1))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    out = cv2.resize(img, (out_w, out_h), interpolation=interp)
    return out, scale


def _pose_label_path(out_dir: Path, *, split: str, key: str) -> Path:
    return out_dir / 'labels_pose' / split / f'{key}.txt'


def _load_existing_kps(label_path: Path, *, n_boxes: int) -> list[list[Kp]]:
    out: list[list[Kp]] = [[Kp(), Kp()] for _ in range(n_boxes)]
    if not label_path.exists():
        return out
    lines = label_path.read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(lines[:n_boxes]):
        parts = line.strip().split()
        if len(parts) != 5 + 2 * 3:
            continue
        k0 = 5
        for kpi in range(2):
            try:
                x = float(parts[k0 + kpi * 3 + 0])
                y = float(parts[k0 + kpi * 3 + 1])
                v = int(float(parts[k0 + kpi * 3 + 2]))
            except ValueError:
                continue
            out[i][kpi] = Kp(x=x, y=y, v=1 if v > 0 else 0)
            if out[i][kpi].v <= 0:
                out[i][kpi].x = 0.0
                out[i][kpi].y = 0.0
    return out


def _write_pose_labels(
    out_path: Path,
    *,
    bboxes: list[YoloBBox],
    kps_by_box: list[list[Kp]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, b in enumerate(bboxes):
        kps = kps_by_box[i] if i < len(kps_by_box) else [Kp(), Kp()]
        parts = [
            str(int(b.cls_id)),
            f'{float(b.cx):.6f}',
            f'{float(b.cy):.6f}',
            f'{float(b.w):.6f}',
            f'{float(b.h):.6f}',
        ]
        for kp in kps[:2]:
            v = 1 if int(kp.v) > 0 else 0
            x = float(kp.x) if v > 0 else 0.0
            y = float(kp.y) if v > 0 else 0.0
            parts.extend([f'{x:.6f}', f'{y:.6f}', str(v)])
        lines.append(' '.join(parts))
    out_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def _point_disp(kp: Kp, *, img_w: int, img_h: int, scale: float) -> tuple[int, int]:
    x = int(round(kp.x * img_w * scale))
    y = int(round(kp.y * img_h * scale))
    return x, y


def _dist2(ax: int, ay: int, bx: int, by: int) -> int:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _crop_xyxy_from_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    img_w: int,
    img_h: int,
    margin: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = float(margin) * bw
    pad_y = float(margin) * bh
    cx1 = _clamp(x1 - pad_x, 0.0, float(img_w - 1))
    cy1 = _clamp(y1 - pad_y, 0.0, float(img_h - 1))
    cx2 = _clamp(x2 + pad_x, 0.0, float(img_w))
    cy2 = _clamp(y2 + pad_y, 0.0, float(img_h))
    ix1 = int(math.floor(cx1))
    iy1 = int(math.floor(cy1))
    ix2 = int(math.ceil(max(cx2, cx1 + 1.0)))
    iy2 = int(math.ceil(max(cy2, cy1 + 1.0)))
    ix1 = max(0, min(img_w - 1, ix1))
    iy1 = max(0, min(img_h - 1, iy1))
    ix2 = max(ix1 + 1, min(img_w, ix2))
    iy2 = max(iy1 + 1, min(img_h, iy2))
    return ix1, iy1, ix2, iy2


def _bbox_xyxy_to_int_clamped(
    bbox_xyxy: tuple[float, float, float, float], *, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = _clamp(x1, 0.0, float(img_w - 1))
    y1 = _clamp(y1, 0.0, float(img_h - 1))
    x2 = _clamp(x2, 0.0, float(img_w))
    y2 = _clamp(y2, 0.0, float(img_h))
    ix1 = int(math.floor(x1))
    iy1 = int(math.floor(y1))
    ix2 = int(math.ceil(max(x2, x1 + 1.0)))
    iy2 = int(math.ceil(max(y2, y1 + 1.0)))
    ix1 = max(0, min(img_w - 1, ix1))
    iy1 = max(0, min(img_h - 1, iy1))
    ix2 = max(ix1 + 1, min(img_w, ix2))
    iy2 = max(iy1 + 1, min(img_h, iy2))
    return ix1, iy1, ix2, iy2


def _sanitize_bbox_xyxy(
    bbox_xyxy: tuple[float, float, float, float], *, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """
    Defensive bbox sanitation for broken labels:
    - clamps to image bounds
    - enforces x2 > x1 and y2 > y1
    """
    x1, y1, x2, y2 = bbox_xyxy
    x1 = _clamp(x1, 0.0, float(img_w - 1))
    y1 = _clamp(y1, 0.0, float(img_h - 1))
    x2 = _clamp(x2, 0.0, float(img_w))
    y2 = _clamp(y2, 0.0, float(img_h))
    if x2 <= x1:
        x2 = min(float(img_w), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(img_h), y1 + 1.0)
    return float(x1), float(y1), float(x2), float(y2)


def _kps_done_for_bbox(kps: list[Kp]) -> bool:
    return any(int(kp.v) > 0 for kp in kps[:2])


def _is_tiny_bbox_xyxy(bbox_xyxy: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    return (x2 - x1) < MIN_BBOX_SIDE_PX or (y2 - y1) < MIN_BBOX_SIDE_PX


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Annotate 2 keypoints on full-frame detection samples (multi-box).\n'
            'BBoxes are read from the detection dataset labels; keypoints are saved to a separate labels_pose folder.\n'
            'Annotation is performed on per-bbox crops (bbox + margin) for easier, stable clicking.\n\n'
            'Mouse:\n'
            '  LMB: click near an existing keypoint to remove it; otherwise set the active (or next missing) keypoint\n'
            f'  Crop is always scaled so the larger side is {DISPLAY_MAX_SIDE}px\n\n'
            'Keys:\n'
            '  Space: accept this bbox; auto-advance to next bbox; on last bbox writes label and advances to next image\n'
            '  p / n: previous / next bbox crop (wraps across images)\n'
            '  1 / 2: choose active keypoint\n'
            '  v: mark active keypoint not visible (sets x=y=0,v=0)\n'
            '  Backspace: delete pose label for this image (resets all bboxes)\n'
            '  Esc: quit'
        )
    )
    p.add_argument('--src', type=Path, required=True, help='Detection dataset root (images + YOLO bbox .txt labels).')
    p.add_argument('--out', type=Path, required=True, help='Pose project output directory.')
    p.add_argument('--val-ratio', type=float, default=0.05, help='Validation split fraction (only used on first run).')
    p.add_argument(
        '--seed', type=int, default=0, help='RNG seed for deterministic split/index (only used on first run).'
    )
    p.add_argument('--hit-radius', type=int, default=18, help='Click radius (px in display coords) to remove a point.')
    p.add_argument('--crop-margin', type=float, default=0.15, help='Crop margin around bbox (fraction of bbox size).')
    p.add_argument(
        '--show-annotated',
        action='store_true',
        help='Include already-annotated samples in navigation (default: only unlabeled).',
    )
    return p.parse_args()


def _index_path(out_dir: Path) -> Path:
    return out_dir / 'pose_index.yaml'


def _load_or_create_index(
    *,
    src_dir: Path,
    out_dir: Path,
    val_ratio: float,
    seed: int,
) -> list[dict]:
    idx_path = _index_path(out_dir)
    if idx_path.exists():
        payload = yaml.safe_load(idx_path.read_text(encoding='utf-8')) or {}
        items = payload.get('items', [])
        if isinstance(items, list):
            return [it for it in items if isinstance(it, dict)]
        return []

    import random

    images = _collect_images(src_dir)
    items: list[dict] = []
    used_keys: dict[str, int] = {}
    for img_path in images:
        label_path = img_path.with_suffix('.txt')
        bboxes = _read_bboxes(label_path)
        if not bboxes:
            continue
        rel = img_path.relative_to(src_dir)
        key = _slug_from_relpath(rel)
        if key in used_keys:
            used_keys[key] += 1
            key = f'{key}__dup{used_keys[key]:02d}'
        else:
            used_keys[key] = 0
        items.append(
            {
                'key': key,
                'src_rel': str(rel.as_posix()),
                'split': 'train',
            }
        )

    if not items:
        raise SystemExit(f'No labeled detections found under: {src_dir}')

    rng = random.Random(int(seed))
    rng.shuffle(items)
    n_val = int(math.floor(len(items) * float(val_ratio)))
    n_val = max(1, n_val) if len(items) >= 2 else 0
    for i, it in enumerate(items):
        it['split'] = 'val' if i < n_val else 'train'

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': 1,
        'src_root': str(src_dir.resolve()),
        'seed': int(seed),
        'val_ratio': float(val_ratio),
        'kpt_names': KP_NAMES,
        'items': items,
    }
    idx_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return items


def main() -> int:
    args = _parse_args()
    src_dir = Path(args.src)
    out_dir = Path(args.out)
    if not src_dir.exists():
        raise SystemExit(f'--src does not exist: {src_dir}')

    items = _load_or_create_index(
        src_dir=src_dir,
        out_dir=out_dir,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    if not items:
        raise SystemExit(f'No index items found in: {_index_path(out_dir)}')

    for sp in ('train', 'val'):
        (out_dir / 'labels_pose' / sp).mkdir(parents=True, exist_ok=True)

    win = 'keypoints-fullframe'
    cv2.namedWindow(win)
    screen_size = get_screen_size()

    state = {
        'idx': 0,
        'bbox_idx': 0,
        'active_kp': None,  # None => next missing
        'click': None,  # (x,y) in display coords
        'hud': None,  # transient message
        'hud_until': 0.0,
    }

    def mouse_cb(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['click'] = (int(x), int(y))

    cv2.setMouseCallback(win, mouse_cb)

    cached = {
        'key': None,
        'img': None,
        'img_w': 0,
        'img_h': 0,
        'bboxes': None,
        'kps_by_box': None,
        'pose_path': None,
        'split': None,
        'src_path': None,
    }

    hit_r2 = int(args.hit_radius) * int(args.hit_radius)

    def pose_path_for(it: dict) -> Path:
        return _pose_label_path(out_dir, split=str(it.get('split', 'train')), key=str(it.get('key', '')))

    def is_annotated(it: dict) -> bool:
        try:
            return pose_path_for(it).exists()
        except Exception:
            return False

    def seek_index(start: int, direction: int) -> Optional[int]:
        start = int(start)
        direction = 1 if int(direction) >= 0 else -1
        if bool(args.show_annotated):
            return max(0, min(start, len(items) - 1))
        i = start
        while 0 <= i < len(items) and is_annotated(items[i]):
            i += direction
        if 0 <= i < len(items):
            return i
        return None

    first = seek_index(0, 1)
    if first is None:
        raise SystemExit('No unlabeled samples found (everything already has pose labels).')
    state['idx'] = int(first)

    while True:
        idx = int(state['idx'])
        idx = max(0, min(idx, len(items) - 1))
        state['idx'] = idx
        item = items[idx]
        key = str(item.get('key', ''))
        split = str(item.get('split', 'train'))
        rel = str(item.get('src_rel', ''))
        src_path = src_dir / Path(rel)
        pose_path = _pose_label_path(out_dir, split=split, key=key)

        if cached['key'] != key:
            img = cv2.imread(str(src_path))
            if img is None:
                nxt = seek_index(idx + 1, 1)
                if nxt is None:
                    break
                state['idx'] = int(nxt)
                cached['key'] = None
                continue
            img_h, img_w = img.shape[:2]
            bboxes = _read_bboxes(src_path.with_suffix('.txt'))
            if not bboxes:
                nxt = seek_index(idx + 1, 1)
                if nxt is None:
                    break
                state['idx'] = int(nxt)
                cached['key'] = None
                continue

            kps_by_box = _load_existing_kps(pose_path, n_boxes=len(bboxes))
            state['bbox_idx'] = max(0, min(int(state['bbox_idx']), len(bboxes) - 1))

            cached.update(
                {
                    'key': key,
                    'img': img,
                    'img_w': int(img_w),
                    'img_h': int(img_h),
                    'bboxes': bboxes,
                    'kps_by_box': kps_by_box,
                    'pose_path': pose_path,
                    'split': split,
                    'src_path': src_path,
                }
            )

        img = cached['img']
        if img is None:
            break
        img_w = int(cached['img_w'])
        img_h = int(cached['img_h'])
        bboxes = cached['bboxes'] or []
        kps_by_box = cached['kps_by_box'] or []
        bbox_idx = max(0, min(int(state['bbox_idx']), len(bboxes) - 1))
        state['bbox_idx'] = bbox_idx

        # Crop view for current bbox (+ margin)
        bbox_xyxy_raw = bboxes[bbox_idx].to_xyxy_abs(img_w=img_w, img_h=img_h)
        bbox_xyxy = _sanitize_bbox_xyxy(bbox_xyxy_raw, img_w=img_w, img_h=img_h)
        if _is_tiny_bbox_xyxy(bbox_xyxy):
            state['hud'] = f'Skipping tiny bbox ({MIN_BBOX_SIDE_PX}px).'
            state['hud_until'] = float(time.time()) + 1.5
            if bbox_idx < len(bboxes) - 1:
                state['bbox_idx'] = bbox_idx + 1
            else:
                nxt = seek_index(idx + 1, 1)
                if nxt is None:
                    break
                state['idx'] = int(nxt)
                cached['key'] = None
                state['bbox_idx'] = 0
            continue
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = _bbox_xyxy_to_int_clamped(bbox_xyxy, img_w=img_w, img_h=img_h)
        crop_x1, crop_y1, crop_x2, crop_y2 = _crop_xyxy_from_bbox(
            bbox_xyxy, img_w=img_w, img_h=img_h, margin=float(args.crop_margin)
        )
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        disp, crop_scale = _resize_to_max_side(crop, DISPLAY_MAX_SIDE)
        base_h, base_w = disp.shape[:2]

        # Handle click
        click = state.get('click', None)
        if click is not None:
            cx, cy = int(click[0]), int(click[1])
            state['click'] = None

            # Remove if near an existing visible kp
            removed = False
            for kpi in range(2):
                kp = kps_by_box[bbox_idx][kpi]
                if kp.v <= 0:
                    continue
                # kp is normalized in full-frame coords; map into crop-display coords
                abs_x = kp.x * img_w
                abs_y = kp.y * img_h
                loc_x = (abs_x - float(crop_x1)) * float(crop_scale)
                loc_y = (abs_y - float(crop_y1)) * float(crop_scale)
                px = int(round(loc_x))
                py = int(round(loc_y))
                if _dist2(int(cx), int(cy), px, py) <= hit_r2:
                    kps_by_box[bbox_idx][kpi] = Kp(0.0, 0.0, 0)
                    removed = True
                    break

            if not removed:
                # Click is in crop-display coords; map back to absolute image pixels.
                abs_x = float(crop_x1) + float(cx) / float(max(1e-6, crop_scale))
                abs_y = float(crop_y1) + float(cy) / float(max(1e-6, crop_scale))
                # Clamp keypoints to the bbox (tight), as requested.
                abs_x = _clamp(abs_x, float(bbox_x1), float(bbox_x2 - 1))
                abs_y = _clamp(abs_y, float(bbox_y1), float(bbox_y2 - 1))
                nx = _clamp(abs_x / float(img_w), 0.0, 1.0)
                ny = _clamp(abs_y / float(img_h), 0.0, 1.0)

                active_kp: Optional[int] = state['active_kp']
                if active_kp is None:
                    active_kp = 0 if kps_by_box[bbox_idx][0].v <= 0 else 1 if kps_by_box[bbox_idx][1].v <= 0 else 0
                kps_by_box[bbox_idx][active_kp] = Kp(nx, ny, 1)

        # Draw UI
        canvas = disp.copy()
        labeled = pose_path.exists()
        title = (
            f'{split.upper()}  {"LABELED" if labeled else "UNLABELED"}  '
            f'{idx + 1}/{len(items)}  {src_path.name}   bbox {bbox_idx + 1}/{len(bboxes)}'
        )
        cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        active_kp = state['active_kp']
        akp_txt = 'auto' if active_kp is None else str(int(active_kp) + 1)
        cv2.putText(
            canvas,
            f'active kp: {akp_txt}',
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Legend
        y = 86
        for i, name in enumerate(KP_NAMES):
            color = KP_COLORS_BGR[i]
            cv2.circle(canvas, (18, y - 6), 7, color, -1)
            cv2.putText(canvas, f'{i + 1}: {name}', (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            y += 26

        # Draw bbox outline inside the crop (for clamping reference)
        bx1 = int(round((float(bbox_x1) - float(crop_x1)) * float(crop_scale)))
        by1 = int(round((float(bbox_y1) - float(crop_y1)) * float(crop_scale)))
        bx2 = int(round((float(bbox_x2) - float(crop_x1)) * float(crop_scale)))
        by2 = int(round((float(bbox_y2) - float(crop_y1)) * float(crop_scale)))
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

        # Draw keypoints
        for kpi in range(2):
            kp = kps_by_box[bbox_idx][kpi]
            if kp.v <= 0:
                continue
            abs_x = kp.x * img_w
            abs_y = kp.y * img_h
            loc_x = (abs_x - float(crop_x1)) * float(crop_scale)
            loc_y = (abs_y - float(crop_y1)) * float(crop_scale)
            px = int(round(loc_x))
            py = int(round(loc_y))
            color = KP_COLORS_BGR[kpi]
            cv2.circle(canvas, (px, py), 7, (0, 0, 0), -1)
            cv2.circle(canvas, (px, py), 5, color, -1)
            cv2.putText(
                canvas, str(kpi + 1), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
            )

        hint = 'LMB set/remove; Space accept+next; p/n prev/next bbox; 1/2 kp; v not visible; Backspace reset image; Esc quit'
        cv2.putText(
            canvas, hint, (10, canvas.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

        # HUD message (transient)
        hud = state.get('hud', None)
        if hud and float(time.time()) < float(state.get('hud_until', 0.0)):
            cv2.putText(
                canvas,
                str(hud),
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        canvas = overlay_screen_warning(canvas, screen_size)
        cv2.imshow(win, canvas)
        keycode = cv2.waitKey(20)

        if keycode == 27:  # Esc
            break

        if keycode == ord('p'):
            if bbox_idx > 0:
                state['bbox_idx'] = bbox_idx - 1
            else:
                prev_idx = seek_index(idx - 1, -1)
                if prev_idx is None:
                    continue
                state['idx'] = int(prev_idx)
                cached['key'] = None
                state['bbox_idx'] = 10**9  # clamped after load
        elif keycode == ord('n'):
            if bbox_idx < len(bboxes) - 1:
                state['bbox_idx'] = bbox_idx + 1
            else:
                next_idx = seek_index(idx + 1, 1)
                if next_idx is None:
                    continue
                state['idx'] = int(next_idx)
                cached['key'] = None
                state['bbox_idx'] = 0
        elif keycode == ord('1'):
            state['active_kp'] = 0
        elif keycode == ord('2'):
            state['active_kp'] = 1
        elif keycode == ord('0'):
            state['active_kp'] = None
        elif keycode == ord('v'):
            ak = state['active_kp']
            if ak is None:
                ak = 0
            kps_by_box[bbox_idx][ak] = Kp(0.0, 0.0, 0)
        elif keycode == 32:  # Space accept bbox and advance
            if not _kps_done_for_bbox(kps_by_box[bbox_idx]):
                state['hud'] = 'This bbox needs at least 1 keypoint.'
                state['hud_until'] = float(time.time()) + 2.0
                continue

            if bbox_idx < len(bboxes) - 1:
                state['bbox_idx'] = bbox_idx + 1
            else:
                # On last bbox: require all bboxes done before writing label.
                incomplete = next((i for i in range(len(bboxes)) if not _kps_done_for_bbox(kps_by_box[i])), None)
                if incomplete is not None:
                    state['hud'] = f'Bbox {incomplete + 1}/{len(bboxes)} still missing keypoints.'
                    state['hud_until'] = float(time.time()) + 2.0
                    state['bbox_idx'] = int(incomplete)
                    continue

                _write_pose_labels(pose_path, bboxes=bboxes, kps_by_box=kps_by_box)
                nxt = seek_index(idx + 1, 1)
                if nxt is None:
                    break
                state['idx'] = int(nxt)
                cached['key'] = None
                state['bbox_idx'] = 0
        elif keycode == 8:  # Backspace delete pose label
            if pose_path.exists():
                pose_path.unlink()
                cached['key'] = None
            # Reset in-memory kps for this image
            for i in range(len(kps_by_box)):
                kps_by_box[i] = [Kp(), Kp()]

    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
