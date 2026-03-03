#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


SUPPORTED_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Promote reviewed pseudo-labeled bbox samples into the main training dataset with collision-safe naming.'
    )
    p.add_argument('--src', type=Path, required=True, help='Reviewed pseudo-labeled folder (image + txt pairs).')
    p.add_argument(
        '--dst', type=Path, default=Path('train/detection/windsurf_dataset'), help='Destination dataset folder.'
    )
    p.add_argument('--copy', dest='copy', action='store_true', help='Copy files (default behavior, non-destructive).')
    p.add_argument('--move', dest='copy', action='store_false', help='Move files instead of copying.')
    p.set_defaults(copy=True)
    return p.parse_args()


def _iter_images(src: Path) -> list[Path]:
    return [p for p in sorted(src.glob('*')) if p.is_file() and p.suffix.lower() in SUPPORTED_IMG_EXTS]


def _validate_yolo_label(label_path: Path) -> bool:
    try:
        lines = label_path.read_text(encoding='utf-8').splitlines()
    except Exception:
        return False

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            return False
        try:
            float(parts[0])
            vals = list(map(float, parts[1:]))
        except ValueError:
            return False
        cx, cy, bw, bh = vals
        if bw <= 0 or bh <= 0:
            return False
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0):
            return False
    return True


def _next_pair_path(dst: Path, stem: str, img_suffix: str) -> tuple[Path, Path, bool]:
    img = dst / f'{stem}{img_suffix}'
    txt = dst / f'{stem}.txt'
    if not img.exists() and not txt.exists():
        return img, txt, False

    bump = 1
    while True:
        img = dst / f'{stem}__dup{bump:02d}{img_suffix}'
        txt = dst / f'{stem}__dup{bump:02d}.txt'
        if not img.exists() and not txt.exists():
            return img, txt, True
        bump += 1


def _transfer(src_file: Path, dst_file: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(src_file, dst_file)
    else:
        shutil.move(str(src_file), str(dst_file))


def main() -> int:
    args = _parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        raise SystemExit(f'--src does not exist: {src}')

    dst.mkdir(parents=True, exist_ok=True)

    images = _iter_images(src)

    counts = {
        'mode': 'copy' if bool(args.copy) else 'move',
        'images_found': len(images),
        'copied': 0,
        'renamed': 0,
        'skipped_missing_label': 0,
        'skipped_invalid_label': 0,
        'skipped_transfer_error': 0,
        'source_dir': str(src.resolve()),
        'destination_dir': str(dst.resolve()),
    }

    for img_path in images:
        txt_path = img_path.with_suffix('.txt')
        if not txt_path.exists():
            counts['skipped_missing_label'] += 1
            continue
        if not _validate_yolo_label(txt_path):
            counts['skipped_invalid_label'] += 1
            continue

        out_img, out_txt, renamed = _next_pair_path(dst, img_path.stem, img_path.suffix.lower())

        try:
            _transfer(img_path, out_img, bool(args.copy))
            _transfer(txt_path, out_txt, bool(args.copy))
        except Exception:
            counts['skipped_transfer_error'] += 1
            # If one file already moved/copied and the second failed, leave it as-is and count error.
            continue

        counts['copied'] += 1
        if renamed:
            counts['renamed'] += 1

    print(yaml.safe_dump(counts, sort_keys=False).strip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
