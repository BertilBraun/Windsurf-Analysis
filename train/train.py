#!/usr/bin/env python3
"""
prepare_and_train_windsurfers.py

1. Re-structure raw annotated frames into the Ultralytics format.
2. Create windsufers.yaml.
3. Fine-tune YOLO v11 in one go.

Usage
-----
python prepare_and_train_windsurfers.py \
    --src ./dataset \
    --dst ./datasets/windsurfers \
    --val-ratio 0.02 \
    --epochs 20 \
    --imgsz 640 \
    --batch 0.7 \
    --device auto

Arguments
---------
--src         Directory holding *.jpg / *.txt pairs from the annotation tool.
--dst         Where the new dataset tree and YAML will be written.
--val-ratio   Fraction of images for validation.
--epochs      Training epochs.
--imgsz       Square input resolution.
--batch       Batch-size fraction (0 = YOLO chooses automatically).
--device      GPU id, -1 for CPU, or "auto".
"""

import argparse
import random
import shutil
import sys
import logging
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def setup_logging():
    """Configure logging for the windsurfer training tool."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('windsurf_training.log')],
    )
    return logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ────────────────────────────────────────────────────────────────────────────
def prepare_dataset(src: Path, dst: Path, val_ratio: float = 0.02, seed: int = 0) -> Path:
    logger = logging.getLogger(__name__)

    if not src.exists():
        logger.error(f'Source directory {src} does not exist.')
        sys.exit(1)

    # Gather *.jpg files that have a matching *.txt
    images = sorted(p for p in src.glob('*.jpg') if (src / f'{p.stem}.txt').exists())
    if not images:
        logger.error('No matching .jpg /.txt pairs found in the source directory.')
        sys.exit(1)

    random.Random(seed).shuffle(images)
    n_val = max(1, int(len(images) * val_ratio))

    splits = {'val': images[:n_val], 'train': images[n_val:]}

    # Build directory tree and copy
    for split in splits:
        (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)
        # clear the directories
        for file in (dst / 'images' / split).glob('*'):
            file.unlink()
        for file in (dst / 'labels' / split).glob('*'):
            file.unlink()

    for split, split_imgs in splits.items():
        for img_path in split_imgs:
            label_path = src / f'{img_path.stem}.txt'
            shutil.copy2(img_path, dst / 'images' / split / img_path.name)
            shutil.copy2(label_path, dst / 'labels' / split / label_path.name)

    # YAML descriptor
    yaml_path = dst / 'windsurfers.yaml'
    yaml_content = {
        'path': str(dst.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': ['windsurfer'],
        'nc': 1,
    }
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(yaml_content, f)

    logger.info(f'✓ Dataset prepared at {dst.resolve()}')
    logger.info(f'✓ YAML written to  {yaml_path.resolve()}')
    logger.info(f'   Train / Val:    {len(splits["train"])} / {len(splits["val"])} images')
    return yaml_path


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────
def train_model(yaml_file: Path, epochs: int, imgsz: int, batch: float, device: str):
    logger = logging.getLogger(__name__)

    model = YOLO('yolo11n.pt')  # choose e.g. yolo11s.pt or yolo11m.pt for bigger models
    logger.info('🚀  Starting training …')

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model.train(
        data=str(yaml_file),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        single_cls=True,
    )
    logger.info('✓ Training finished')


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def main():
    logger = setup_logging()

    parser = argparse.ArgumentParser(description='Prepare windsurfer detection dataset and train YOLO v11.')
    parser.add_argument('--src', type=Path, help='Raw dataset directory')
    parser.add_argument('--dst', type=Path, default=Path('./datasets/windsurfers'), help='Output dataset root')
    parser.add_argument('--val-ratio', type=float, default=0.02, help='Validation split fraction')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for splitting')

    # training hyper-parameters
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=float, default=0.7, help='Batch-size fraction (0 = auto)')
    parser.add_argument('--device', default='auto', help='GPU id, -1 for CPU, or "auto"')

    args = parser.parse_args()

    yaml_path = prepare_dataset(args.src, args.dst, args.val_ratio, args.seed)
    train_model(yaml_path, args.epochs, args.imgsz, args.batch, args.device)


if __name__ == '__main__':
    main()
