#!/usr/bin/env python3
"""
train_orientation_end2end.py

End-to-end trainer for a 4-class (0/90/180/270) orientation classifier:
  - Builds a temporary YOLOv8-classification dataset from upright (0°) videos.
  - Trains a YOLOv8 classification model with only the *necessary* augmentation overrides:
        degrees=0.0, fliplr=0.0, flipud=0.0, auto_augment="none"
    (Everything else remains at Ultralytics defaults.)
  - Saves weights to --outdir and then deletes the temporary dataset.

Usage example:
  pip install ultralytics opencv-python tqdm
  python train_orientation_end2end.py \
      --videos ./videos_upright \
      --outdir ./orientation_runs \
      --sample-prob 0.10 \
      --balance \
      --epochs 30 --batch 64 --imgsz 320

Notes:
- Input videos should be upright (0°).
- We randomly select ~p% of frames; each selected frame gets exactly ONE label: 0/90/180/270.
- No split leakage by construction (each base frame appears once).
"""

import argparse
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO

DEGREES = (0, 90, 180, 270)


# -----------------------------
# Dataset building
# -----------------------------
def is_video(p: Path) -> bool:
    return p.suffix.lower() in {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}


def ensure_layout(root: Path):
    for split in ('train', 'val', 'test'):
        for d in DEGREES:
            (root / split / str(d)).mkdir(parents=True, exist_ok=True)


def rotate_deg(img_rgb: np.ndarray, deg: int) -> np.ndarray:
    d = deg % 360
    if d == 0:
        return img_rgb
    if d == 90:
        return cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(img_rgb, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError('deg must be one of {0,90,180,270}')


def pick_split(rng: random.Random, train_ratio: float, val_ratio: float) -> str:
    r = rng.random()
    if r < train_ratio:
        return 'train'
    elif r < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'


def pick_degree(rng: random.Random, counts: Dict[int, int], balance: bool) -> int:
    if not balance:
        return rng.choice(DEGREES)
    m = min(counts[d] for d in DEGREES)
    under = [d for d in DEGREES if counts[d] == m]
    return rng.choice(under)


def build_dataset(
    videos_dir: Path,
    dataset_root: Path,
    sample_prob: float,
    train_ratio: float,
    val_ratio: float,
    jpg_quality: int,
    balance: bool,
    seed: int,
) -> dict:
    """
    Builds classification dataset in one pass over each video.
    Returns a summary dict with per-split and per-class counts.
    """
    rng = random.Random(seed)
    ensure_layout(dataset_root)

    deg_counts = {d: 0 for d in DEGREES}
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    total = 0

    videos = [p for p in sorted(videos_dir.iterdir()) if p.is_file() and is_video(p)]
    if not videos:
        raise SystemExit(f"No videos found in '{videos_dir}'")

    print(f'[info] Found {len(videos)} videos.')
    for vid in tqdm(videos, desc='Building dataset'):
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f'[warn] Could not open {vid}')
            continue

        base = vid.stem
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if rng.random() < sample_prob:
                split = pick_split(rng, train_ratio, val_ratio)
                deg = pick_degree(rng, deg_counts, balance)

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_rot = rotate_deg(rgb, deg)
                bgr_out = cv2.cvtColor(rgb_rot, cv2.COLOR_RGB2BGR)

                out_dir = dataset_root / split / str(deg)
                out_name = f'{base}_f{idx:07d}_{deg}.jpg'
                cv2.imwrite(str(out_dir / out_name), bgr_out, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

                deg_counts[deg] += 1
                split_counts[split] += 1
                total += 1

            idx += 1

        cap.release()

    if total == 0:
        raise SystemExit('No images were written. Consider increasing --sample-prob or check that videos can be read.')

    summary = {
        'total_images': total,
        'per_split': split_counts,
        'per_class': deg_counts,
    }

    # Persist a little README/summary inside the dataset dir (helpful if you keep it)
    (dataset_root / 'README.txt').write_text(f'YOLOv8 classification dataset for 0/90/180/270.\n{summary}\n')

    print('\n[dataset summary]')
    print(' total images:', total)
    print(' per-split   :', split_counts)
    print(' per-class   :', deg_counts)
    return summary


# -------------
# Train helper
# -------------
def train_yolov8_cls(
    dataset_root: Path,
    outdir: Path,
    run_name: str,
    weights: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str | None,
):
    model = YOLO(weights)  # e.g., "yolov8n-cls.pt" or "yolo11n-cls.pt"

    model.train(
        data=str(dataset_root),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device if device else None,
        project=str(outdir),
        name=run_name,
        # ---- NECESSARY overrides only ----
        degrees=0.0,  # no random rotation
        fliplr=0.0,  # no left-right flip
        flipud=0.0,  # no upside-down flip
        auto_augment=None,  # disable policies that may include flips/rotations
        # (hsv/translate/scale/shear/perspective/erasing etc. remain at defaults)
    )

    print('[done] Training complete.')


# -------------
# CLI / main
# -------------
def parse_args():
    ap = argparse.ArgumentParser(
        description='End-to-end orientation classifier training (dataset build + train + cleanup).'
    )
    ap.add_argument('--videos', required=True, type=Path, help='Folder with upright (0°) videos.')
    ap.add_argument('--outdir', default='runs', type=Path, help='Folder to save training runs/weights.')
    ap.add_argument('--run-name', default='yolov8-cls-orientation', help='Name for this training run.')
    # Dataset sampling
    ap.add_argument('--sample-prob', type=float, default=0.05, help='Probability to keep any frame, e.g. 0.05 = 5%.')
    ap.add_argument('--train-ratio', type=float, default=0.95, help='Train split ratio.')
    ap.add_argument('--val-ratio', type=float, default=0.1, help='Val split ratio (rest goes to test).')
    ap.add_argument('--jpg-quality', type=int, default=92)
    ap.add_argument('--balance', action='store_true', help='Balance class counts as you sample.')
    ap.add_argument('--seed', type=int, default=1234)
    # Training
    ap.add_argument(
        '--weights', default='yolov8n-cls.pt', help='Init checkpoint, e.g. yolov8n-cls.pt or yolo11n-cls.pt'
    )
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--device', default=None, help="CUDA device like '0' or '0,1'; leave empty for auto.")
    # Cleanup
    ap.add_argument('--keep-dataset', action='store_true', help='Do NOT delete the temporary dataset (for debugging).')
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Prepare output dir
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Temporary dataset root (auto-removed unless --keep-dataset)
    tmp_root = Path(tempfile.mkdtemp(prefix='orient_ds_'))

    try:
        # 1) Build dataset
        build_dataset(
            videos_dir=args.videos,
            dataset_root=tmp_root,
            sample_prob=args.sample_prob,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            jpg_quality=args.jpg_quality,
            balance=args.balance,
            seed=args.seed,
        )

        # 2) Train (weights saved under --outdir / --run-name)
        train_yolov8_cls(
            dataset_root=tmp_root,
            outdir=args.outdir,
            run_name=args.run_name,
            weights=args.weights,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )

        # 3) Record a small summary alongside weights
        summary_txt = args.outdir / args.run_name / 'dataset_summary.txt'
        if tmp_root.joinpath('README.txt').exists():
            shutil.copy(tmp_root / 'README.txt', summary_txt)

    finally:
        # 4) Cleanup dataset unless user asked to keep it
        if args.keep_dataset:
            print(f'[info] Keeping temporary dataset at: {tmp_root}')
        else:
            try:
                shutil.rmtree(tmp_root)
                print(f'[cleanup] Deleted temporary dataset: {tmp_root}')
            except Exception as e:
                print(f'[warn] Failed to delete temporary dataset {tmp_root}: {e}')


if __name__ == '__main__':
    main()
