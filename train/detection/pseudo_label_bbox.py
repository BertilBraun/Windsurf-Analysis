#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO


VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv'}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Generate pseudo-labeled bbox samples from videos using a trained detection model. '\
            'Writes image + YOLO txt pairs to an output review folder.'
        )
    )
    p.add_argument('--videos', type=Path, required=True, help='Video root folder (recursively scanned).')
    p.add_argument('--model', type=Path, required=True, help='Trained detection model checkpoint (best.pt).')
    p.add_argument('--out', type=Path, required=True, help='Output folder for pseudo-labeled samples.')

    p.add_argument('--max-samples', type=int, default=300, help='Maximum number of sampled frames to process.')
    p.add_argument('--sample-stride', type=int, default=45, help='Frame stride when building frame candidates.')
    p.add_argument('--predict-batch', type=int, default=8, help='Inference batch size (frames per batch).')
    p.add_argument('--conf', type=float, default=0.45, help='Minimum detection confidence.')
    p.add_argument('--iou-nms', type=float, default=0.60, help='NMS IoU threshold for model prediction.')
    p.add_argument('--min-box-side', type=int, default=12, help='Minimum accepted box side length in pixels.')
    p.add_argument('--max-boxes-per-image', type=int, default=3, help='Maximum boxes kept per image (top confidence).')
    p.add_argument(
        '--edge-margin-frac',
        type=float,
        default=0.01,
        help='Reject boxes that touch image edges within this fractional margin.',
    )
    p.add_argument('--device', type=str, default='auto', help='Inference device: auto, cpu, cuda, or id string.')
    p.add_argument('--seed', type=int, default=0, help='RNG seed for deterministic frame selection.')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output image/label files.')
    return p.parse_args()


def _sanitize_stem(name: str) -> str:
    clean = re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('._-')
    return clean or 'video'


def _collect_videos(videos_root: Path) -> list[Path]:
    return [p for p in sorted(videos_root.rglob('*')) if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def _collect_frame_candidates(videos: list[Path], stride: int) -> list[tuple[Path, int]]:
    candidates: list[tuple[Path, int]] = []
    stride = max(1, int(stride))
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        fcnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fcnt <= 0:
            continue
        for fi in range(0, fcnt, stride):
            candidates.append((v, fi))
    return candidates


def _choose_device(device: str) -> str:
    d = str(device)
    if d != 'auto':
        return d
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def _is_near_edge(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    edge_margin_frac: float,
) -> bool:
    mx = float(edge_margin_frac) * float(width)
    my = float(edge_margin_frac) * float(height)
    return x1 <= mx or y1 <= my or x2 >= (float(width) - mx) or y2 >= (float(height) - my)


def _to_yolo_line(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))

    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + (bw / 2.0)
    cy = y1 + (bh / 2.0)
    return f'0 {cx / width:.6f} {cy / height:.6f} {bw / width:.6f} {bh / height:.6f}'


def _batched(items: list, batch_size: int) -> list[list]:
    batch_size = max(1, int(batch_size))
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _next_output_paths(out_dir: Path, stem: str, allow_overwrite: bool) -> tuple[Path, Path, bool]:
    img_path = out_dir / f'{stem}.jpg'
    txt_path = out_dir / f'{stem}.txt'
    if allow_overwrite or (not img_path.exists() and not txt_path.exists()):
        return img_path, txt_path, False

    bump = 1
    while True:
        img_path = out_dir / f'{stem}__dup{bump:02d}.jpg'
        txt_path = out_dir / f'{stem}__dup{bump:02d}.txt'
        if not img_path.exists() and not txt_path.exists():
            return img_path, txt_path, True
        bump += 1


def main() -> int:
    args = _parse_args()

    videos_root = Path(args.videos)
    model_path = Path(args.model)
    out_dir = Path(args.out)

    if not videos_root.exists():
        raise SystemExit(f'--videos does not exist: {videos_root}')
    if not model_path.exists():
        raise SystemExit(f'--model does not exist: {model_path}')

    out_dir.mkdir(parents=True, exist_ok=True)

    videos = _collect_videos(videos_root)
    if not videos:
        raise SystemExit(f'No videos found under: {videos_root}')

    candidates = _collect_frame_candidates(videos, int(args.sample_stride))
    if not candidates:
        raise SystemExit('No frame candidates generated (video decode/frame count issue).')

    rng = random.Random(int(args.seed))
    rng.shuffle(candidates)
    max_samples = max(1, int(args.max_samples))
    selected = candidates[:max_samples]

    device = _choose_device(str(args.device))
    model = YOLO(str(model_path))

    counts = {
        'videos_found': len(videos),
        'candidates_total': len(candidates),
        'frames_selected': len(selected),
        'frames_decoded': 0,
        'frames_written': 0,
        'labels_written': 0,
        'renamed_on_collision': 0,
        'skipped_decode_fail': 0,
        'skipped_no_predictions': 0,
        'skipped_all_filtered': 0,
        'filtered_low_conf': 0,
        'filtered_small_box': 0,
        'filtered_edge_margin': 0,
        'output_dir': str(out_dir.resolve()),
    }

    for batch in _batched(selected, int(args.predict_batch)):
        batch_frames: list[dict] = []
        for video_path, frame_idx in batch:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                counts['skipped_decode_fail'] += 1
                continue

            counts['frames_decoded'] += 1
            batch_frames.append({'video': video_path, 'frame_idx': int(frame_idx), 'frame': frame})

        if not batch_frames:
            continue

        sources = [entry['frame'] for entry in batch_frames]
        results = model.predict(
            source=sources,
            conf=float(args.conf),
            iou=float(args.iou_nms),
            device=device,
            verbose=False,
            save=False,
            save_txt=False,
        )

        for entry, res in zip(batch_frames, results):
            frame = entry['frame']
            video_path = Path(entry['video'])
            frame_idx = int(entry['frame_idx'])
            height, width = frame.shape[:2]

            boxes = getattr(res, 'boxes', None)
            if boxes is None or len(boxes) == 0:
                counts['skipped_no_predictions'] += 1
                continue

            xyxy_t = getattr(boxes, 'xyxy', None)
            conf_t = getattr(boxes, 'conf', None)
            if xyxy_t is None or conf_t is None:
                counts['skipped_no_predictions'] += 1
                continue

            pred_xyxy = xyxy_t.detach().cpu().numpy().tolist()
            pred_conf = conf_t.detach().cpu().numpy().tolist()

            kept: list[tuple[float, float, float, float, float]] = []
            for pbox, pconf in zip(pred_xyxy, pred_conf):
                conf_val = float(pconf)
                if conf_val < float(args.conf):
                    counts['filtered_low_conf'] += 1
                    continue
                x1, y1, x2, y2 = map(float, pbox[:4])
                bw = x2 - x1
                bh = y2 - y1
                if bw < float(args.min_box_side) or bh < float(args.min_box_side):
                    counts['filtered_small_box'] += 1
                    continue
                if _is_near_edge(x1, y1, x2, y2, width, height, float(args.edge_margin_frac)):
                    counts['filtered_edge_margin'] += 1
                    continue
                kept.append((conf_val, x1, y1, x2, y2))

            if not kept:
                counts['skipped_all_filtered'] += 1
                continue

            kept.sort(key=lambda it: it[0], reverse=True)
            kept = kept[: max(1, int(args.max_boxes_per_image))]

            yolo_lines = [_to_yolo_line(x1, y1, x2, y2, width, height) for _, x1, y1, x2, y2 in kept]
            if not yolo_lines:
                counts['skipped_all_filtered'] += 1
                continue

            stem = f"{_sanitize_stem(video_path.stem)}_frame_{frame_idx:06d}"
            out_img, out_lbl, renamed = _next_output_paths(out_dir, stem, bool(args.overwrite))

            ok = cv2.imwrite(str(out_img), frame)
            if not ok:
                counts['skipped_decode_fail'] += 1
                continue

            out_lbl.write_text('\n'.join(yolo_lines) + '\n', encoding='utf-8')
            counts['frames_written'] += 1
            counts['labels_written'] += len(yolo_lines)
            if renamed:
                counts['renamed_on_collision'] += 1

    print(yaml.safe_dump(counts, sort_keys=False).strip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
