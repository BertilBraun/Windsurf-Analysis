#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


SUPPORTED_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
KP_NAMES = ['boom_mast', 'mast_tip']
KP_COLORS_BGR = [
    (0, 165, 255),  # orange
    (255, 0, 255),  # magenta
]


@dataclass(frozen=True)
class Sample:
    name: str
    image_bgr: np.ndarray


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Compare multiple YOLO-pose checkpoints by running inference on the same samples and writing a grid image.\n'
            'Samples can come from an images folder or from randomly sampled frames of a video.\n'
            'Only the predicted detection crop is rendered (not the full frame).'
        )
    )
    model_g = p.add_mutually_exclusive_group(required=True)
    model_g.add_argument('--models', nargs='+', type=Path, help='Explicit list of model weights (.pt).')
    model_g.add_argument(
        '--run',
        type=Path,
        help='Ultralytics run folder containing weights/epoch*.pt (use together with --epochs).',
    )
    p.add_argument('--epochs', nargs='+', type=int, default=None, help='Epoch numbers to load from --run/weights.')

    src_g = p.add_mutually_exclusive_group(required=True)
    src_g.add_argument('--images', type=Path, help='Directory of images to sample from.')
    src_g.add_argument('--video', type=Path, help='Video path to sample frames from.')

    p.add_argument('--out', type=Path, required=True, help='Output directory to write the grid image.')
    p.add_argument('--name', type=str, default='checkpoint_grid', help='Output filename stem.')
    p.add_argument('--num-samples', type=int, default=10, help='Number of images/frames to sample.')
    p.add_argument('--seed', type=int, default=0, help='RNG seed for sampling.')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--crop-margin', type=float, default=0.15, help='Crop padding around predicted bbox (fraction).')
    p.add_argument('--tile', type=int, default=320, help='Tile size (square) for each crop render.')
    p.add_argument('--pad', type=int, default=8, help='Padding between tiles in the output grid.')
    p.add_argument('--save-tiles', action='store_true', help='Also save individual tiles per sample/model.')
    return p.parse_args()


def _collect_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_IMG_EXTS:
            continue
        out.append(p)
    return out


def _sample_images(images_dir: Path, *, n: int, seed: int) -> list[Sample]:
    paths = _collect_images(images_dir)
    if not paths:
        raise SystemExit(f'No images found under: {images_dir}')
    rng = random.Random(int(seed))
    if n > 0 and len(paths) > n:
        paths = rng.sample(paths, int(n))
    out: list[Sample] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        out.append(Sample(name=p.stem, image_bgr=img))
    if not out:
        raise SystemExit('Failed to load any images.')
    return out


def _sample_video_frames(video_path: Path, *, n: int, seed: int) -> list[Sample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f'Failed to open video: {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise SystemExit(f'Video reports invalid frame count: {video_path}')

    rng = random.Random(int(seed))
    n = max(1, int(n))
    idxs = [rng.randint(0, total - 1) for _ in range(n)]

    out: list[Sample] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out.append(Sample(name=f'{video_path.stem}_frame_{int(i):06d}', image_bgr=frame))
    cap.release()
    if not out:
        raise SystemExit('Failed to sample any frames from the video.')
    return out


def _resolve_models(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    if args.models:
        paths = [Path(p) for p in args.models]
        for p in paths:
            if not p.exists():
                raise SystemExit(f'Model not found: {p}')
        labels = [p.stem for p in paths]
        return paths, labels

    run = Path(args.run)
    weights_dir = run / 'weights'
    if not weights_dir.exists():
        raise SystemExit(f'Missing weights dir: {weights_dir}')
    epochs = args.epochs or []
    if not epochs:
        # Try to auto-discover epoch checkpoints
        pts = sorted(weights_dir.glob('epoch*.pt'))
        if not pts:
            raise SystemExit(f'No epoch checkpoints found under: {weights_dir}')
        paths = pts
        labels = [p.stem for p in paths]
        return paths, labels

    paths: list[Path] = []
    labels: list[str] = []
    missing: list[str] = []
    for e in epochs:
        p = weights_dir / f'epoch{int(e)}.pt'
        if p.exists():
            paths.append(p)
            labels.append(f'epoch{int(e)}')
        else:
            missing.append(str(p))
    if missing:
        avail = [p.name for p in sorted(weights_dir.glob('epoch*.pt'))]
        raise SystemExit(
            'Missing checkpoints:\n  ' + '\n  '.join(missing) + '\nAvailable:\n  ' + '\n  '.join(avail[:50])
        )
    return paths, labels


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(int(v), hi)))


def _render_crop_tile(
    *,
    img_bgr: np.ndarray,
    model: YOLO,
    label: str,
    device: str,
    imgsz: int,
    conf: float,
    crop_margin: float,
    tile: int,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    tile = max(64, int(tile))
    canvas = np.full((tile, tile, 3), 32, dtype=np.uint8)

    results = model.predict(
        source=img_bgr,
        imgsz=int(imgsz),
        conf=float(conf),
        device=str(device),
        verbose=False,
    )
    if not results:
        cv2.putText(canvas, f'{label}: no result', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return canvas
    r = results[0]
    boxes = getattr(r, 'boxes', None)
    kps = getattr(r, 'keypoints', None)
    if boxes is None or getattr(boxes, 'xyxy', None) is None or getattr(boxes, 'conf', None) is None:
        cv2.putText(canvas, f'{label}: no det', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return canvas

    try:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
    except Exception:
        cv2.putText(canvas, f'{label}: det err', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return canvas
    if xyxy.shape[0] == 0:
        cv2.putText(canvas, f'{label}: no det', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return canvas

    best_i = int(np.argmax(confs))
    x1, y1, x2, y2 = map(float, xyxy[best_i].tolist())
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * float(crop_margin)
    pad_y = bh * float(crop_margin)
    cx1 = _clamp_int(x1 - pad_x, 0, w - 1)
    cy1 = _clamp_int(y1 - pad_y, 0, h - 1)
    cx2 = _clamp_int(x2 + pad_x, cx1 + 1, w)
    cy2 = _clamp_int(y2 + pad_y, cy1 + 1, h)

    crop = img_bgr[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        cv2.putText(canvas, f'{label}: empty crop', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return canvas

    # Resize crop to tile (preserve aspect by letterboxing).
    ch, cw = crop.shape[:2]
    scale = float(tile) / float(max(ch, cw))
    out_w = max(1, int(round(cw * scale)))
    out_h = max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (out_w, out_h), interpolation=interp)
    x_off = (tile - out_w) // 2
    y_off = (tile - out_h) // 2
    canvas[y_off : y_off + out_h, x_off : x_off + out_w] = resized

    # Draw bbox (pred) in crop coords
    bx1 = (x1 - float(cx1)) * scale + x_off
    by1 = (y1 - float(cy1)) * scale + y_off
    bx2 = (x2 - float(cx1)) * scale + x_off
    by2 = (y2 - float(cy1)) * scale + y_off
    cv2.rectangle(
        canvas,
        (int(round(bx1)), int(round(by1))),
        (int(round(bx2)), int(round(by2))),
        (0, 255, 255),
        2,
    )

    # Draw keypoints (typed 2 keypoints expected)
    if kps is not None and getattr(kps, 'xy', None) is not None:
        try:
            kxy = kps.xy.cpu().numpy()
            kconf = kps.conf.cpu().numpy() if getattr(kps, 'conf', None) is not None else None
        except Exception:
            kxy = None
            kconf = None
        if kxy is not None and kxy.shape[0] > best_i:
            pts = kxy[best_i]
            for ki in range(min(2, pts.shape[0])):
                px, py = float(pts[ki][0]), float(pts[ki][1])
                if kconf is not None and float(kconf[best_i][ki]) <= 0:
                    continue
                tx = (px - float(cx1)) * scale + x_off
                ty = (py - float(cy1)) * scale + y_off
                color = KP_COLORS_BGR[ki]
                cv2.circle(canvas, (int(round(tx)), int(round(ty))), 7, (0, 0, 0), -1)
                cv2.circle(canvas, (int(round(tx)), int(round(ty))), 5, color, -1)
                cv2.putText(
                    canvas,
                    str(ki + 1),
                    (int(round(tx)) + 8, int(round(ty)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    # Header text
    txt = f'{label}  conf={float(confs[best_i]):.2f}'
    cv2.putText(canvas, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _make_grid(tiles: list[list[np.ndarray]], *, pad: int, bg: int = 0) -> np.ndarray:
    rows = len(tiles)
    cols = max((len(r) for r in tiles), default=0)
    if rows == 0 or cols == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    tile_h, tile_w = tiles[0][0].shape[:2]
    pad = max(0, int(pad))
    out_h = rows * tile_h + (rows + 1) * pad
    out_w = cols * tile_w + (cols + 1) * pad
    out = np.full((out_h, out_w, 3), int(bg), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            t = tiles[r][c]
            y1 = pad + r * (tile_h + pad)
            x1 = pad + c * (tile_w + pad)
            out[y1 : y1 + tile_h, x1 : x1 + tile_w] = t
    return out


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_paths, labels = _resolve_models(args)
    models = [YOLO(str(p)) for p in model_paths]

    if args.images:
        samples = _sample_images(Path(args.images), n=int(args.num_samples), seed=int(args.seed))
    else:
        samples = _sample_video_frames(Path(args.video), n=int(args.num_samples), seed=int(args.seed))

    tiles: list[list[np.ndarray]] = []
    for s in samples:
        row: list[np.ndarray] = []
        for model, label in zip(models, labels):
            tile = _render_crop_tile(
                img_bgr=s.image_bgr,
                model=model,
                label=label,
                device=str(args.device),
                imgsz=int(args.imgsz),
                conf=float(args.conf),
                crop_margin=float(args.crop_margin),
                tile=int(args.tile),
            )
            if bool(args.save_tiles):
                tile_path = out_dir / f'{args.name}__{s.name}__{label}.jpg'
                cv2.imwrite(str(tile_path), tile)
            row.append(tile)
        tiles.append(row)

    grid = _make_grid(tiles, pad=int(args.pad), bg=0)

    # Add row labels (sample names) at left margin by extending canvas.
    label_w = 420
    labeled = np.full((grid.shape[0], grid.shape[1] + label_w, 3), 0, dtype=np.uint8)
    labeled[:, label_w:, :] = grid
    tile_h = tiles[0][0].shape[0] if tiles and tiles[0] else 0
    pad = int(args.pad)
    for r, s in enumerate(samples):
        y = pad + r * (tile_h + pad) + 26
        cv2.putText(labeled, s.name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    out_file_name: Path = args.video if not args.images else args.images
    out_path = out_dir / f'{out_file_name.stem}.jpg'
    cv2.imwrite(str(out_path), labeled)
    print(f'Wrote: {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
