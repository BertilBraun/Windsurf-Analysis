#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


KP_NAMES = ["boom_mast", "mast_tip"]
SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a YOLO-pose dataset from a pose project and train a pose model.")
    p.add_argument("--src", type=Path, required=True, help="Detection dataset root (images + bbox labels).")
    p.add_argument("--pose", type=Path, required=True, help="Pose project directory (pose_index.yaml + labels_pose/...).")
    p.add_argument("--dst", type=Path, default=Path("train/detection/datasets/windsurfer_pose"), help="Temp dataset root.")
    p.add_argument("--base-model", type=str, default="yolo11n-pose.pt", help="Base pose model weights.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (0 disables). If unset, Ultralytics default is used.",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=float, default=0.85)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--project", type=str, default="train/detection/runs")
    p.add_argument("--name", type=str, default="pose")
    p.add_argument(
        "--save-period",
        type=int,
        default=50,
        help="Save a checkpoint every N epochs (Ultralytics save_period). Set 0 to disable.",
    )
    p.add_argument(
        "--include-pseudo",
        action="store_true",
        help="Include pseudo labels from <pose>/labels_pose_pseudo/{train,val}/ (copied with filename prefix pseudo_).",
    )
    p.add_argument(
        "--pseudo-frac",
        type=float,
        default=1.0,
        help="Fraction of pseudo-labeled samples to include when --include-pseudo (0..1).",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed used when sampling pseudo labels via --pseudo-frac.")
    p.add_argument("--keep-dst", action="store_true", help="Do not delete --dst after training.")
    return p.parse_args()


def _index_path(pose_dir: Path) -> Path:
    return pose_dir / "pose_index.yaml"


def _load_index(pose_dir: Path) -> dict:
    idx_path = _index_path(pose_dir)
    if not idx_path.exists():
        raise SystemExit(f"Missing pose index: {idx_path}")
    payload = yaml.safe_load(idx_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid pose index (expected YAML dict): {idx_path}")
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        raise SystemExit(f"No items found in pose index: {idx_path}")
    return payload


def _find_image(src_dir: Path, rel_posix: str) -> Path:
    p = src_dir / Path(rel_posix)
    if p.exists():
        return p
    raise FileNotFoundError(str(p))


def _ensure_empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for p in path.glob("*"):
        if p.is_dir():
            raise RuntimeError(f"Refusing to clear directory containing subdir: {p}")
        p.unlink()


def _write_dataset_yaml(dst_root: Path) -> Path:
    yaml_path = dst_root / "dataset.yaml"
    payload = {
        "path": str(dst_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": ["windsurfer"],
        "nc": 1,
        "kpt_shape": [2, 3],
        "flip_idx": [0, 1],
        "kpt_names": KP_NAMES,
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> int:
    args = _parse_args()
    src_dir = Path(args.src)
    pose_dir = Path(args.pose)
    dst_root = Path(args.dst)
    if not src_dir.exists():
        raise SystemExit(f"--src does not exist: {src_dir}")
    if not pose_dir.exists():
        raise SystemExit(f"--pose does not exist: {pose_dir}")

    index = _load_index(pose_dir)
    items = index.get("items", [])
    assert isinstance(items, list)

    labels_pose_manual = pose_dir / "labels_pose"
    if not labels_pose_manual.exists():
        raise SystemExit(f"Missing labels_pose dir: {labels_pose_manual}")
    labels_pose_pseudo = pose_dir / "labels_pose_pseudo"
    include_pseudo = bool(args.include_pseudo) and labels_pose_pseudo.exists()

    images_train = dst_root / "images" / "train"
    images_val = dst_root / "images" / "val"
    labels_train = dst_root / "labels" / "train"
    labels_val = dst_root / "labels" / "val"
    for d in [images_train, images_val, labels_train, labels_val]:
        _ensure_empty_dir(d)

    train_count = 0
    val_count = 0
    skipped = 0
    pseudo_included = 0
    manual_included = 0

    import random

    rng = random.Random(int(args.seed))
    pseudo_frac = float(args.pseudo_frac)
    pseudo_frac = max(0.0, min(1.0, pseudo_frac))

    for it in items:
        if not isinstance(it, dict):
            continue
        key = str(it.get("key", ""))
        split = str(it.get("split", "train"))
        rel = str(it.get("src_rel", ""))
        if not key or split not in ("train", "val") or not rel:
            continue
        manual_label_path = labels_pose_manual / split / f"{key}.txt"
        pseudo_label_path = labels_pose_pseudo / split / f"{key}.txt"
        is_pseudo = False
        pose_label_path: Path

        if manual_label_path.exists():
            pose_label_path = manual_label_path
            is_pseudo = False
        elif include_pseudo and pseudo_label_path.exists():
            if pseudo_frac < 1.0 and rng.random() > pseudo_frac:
                skipped += 1
                continue
            pose_label_path = pseudo_label_path
            is_pseudo = True
        else:
            skipped += 1
            continue
        try:
            src_img = _find_image(src_dir, rel)
        except FileNotFoundError:
            skipped += 1
            continue
        if src_img.suffix.lower() not in SUPPORTED_IMG_EXTS:
            skipped += 1
            continue

        out_key = f"pseudo_{key}" if is_pseudo else key
        out_img = (images_train if split == "train" else images_val) / f"{out_key}{src_img.suffix.lower()}"
        out_lbl = (labels_train if split == "train" else labels_val) / f"{out_key}.txt"
        shutil.copy2(src_img, out_img)
        shutil.copy2(pose_label_path, out_lbl)
        if is_pseudo:
            pseudo_included += 1
        else:
            manual_included += 1
        if split == "train":
            train_count += 1
        else:
            val_count += 1

    if train_count == 0:
        raise SystemExit("No labeled TRAIN pose samples found (labels_pose/train/*.txt).")
    if val_count == 0:
        raise SystemExit("No labeled VAL pose samples found (labels_pose/val/*.txt).")

    data_yaml = _write_dataset_yaml(dst_root)

    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = YOLO(str(args.base_model))
    batch_val = float(args.batch)
    batch: int | float
    if batch_val >= 1.0 and abs(batch_val - round(batch_val)) < 1e-9:
        batch = int(round(batch_val))
    else:
        batch = batch_val

    model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        patience=int(args.patience),
        imgsz=int(args.imgsz),
        batch=batch,
        device=device,
        single_cls=True,
        project=str(args.project),
        name=str(args.name),
        save_period=int(args.save_period),
    )

    print(
        yaml.safe_dump(
            {
                "train": train_count,
                "val": val_count,
                "manual_included": manual_included,
                "pseudo_included": pseudo_included,
                "skipped": skipped,
                "dataset_yaml": str(data_yaml),
            },
            sort_keys=False,
        ).strip()
    )

    if not bool(args.keep_dst):
        shutil.rmtree(dst_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
