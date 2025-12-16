from __future__ import annotations

import os
import sys
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm.auto import tqdm
import optuna


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Ensure ultralytics config dir is the repo one (weights/settings)
server_root = project_root
os.environ.setdefault('YOLO_CONFIG_DIR', str(server_root / 'ultralytics'))

from ultralytics import YOLO  # type: ignore

from inference.src.settings import YOLO_MODEL_PATH, DETECTOR_BATCH_SIZE


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _list_images(dataset_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(dataset_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def _load_yolo_labels(txt_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO-format labels and convert to xyxy in pixel coords.

    Format per line: cls x_center y_center width height (all normalized 0..1)
    Returns array shape (N, 4) with [x1, y1, x2, y2]. If file missing/empty -> (0,4).
    """
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes: List[List[float]] = []
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    # cls = int(parts[0])
                    xc = float(parts[1]) * img_w
                    yc = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h
                    x1 = xc - w / 2.0
                    y1 = yc - h / 2.0
                    x2 = xc + w / 2.0
                    y2 = yc + h / 2.0
                    boxes.append([x1, y1, x2, y2])
                except Exception:
                    continue
    except Exception:
        return np.zeros((0, 4), dtype=np.float32)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two [N,4] and [M,4] arrays of xyxy boxes."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0:1].T, b[:, 1:2].T, b[:, 2:3].T, b[:, 3:4].T
    inter_w = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    inter_h = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = inter_w * inter_h
    area_a = np.maximum(0.0, (ax2 - ax1)) * np.maximum(0.0, (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def _match_tp_fp_fn(pred: np.ndarray, gt: np.ndarray, match_iou: float) -> Tuple[int, int, int, float]:
    if len(pred) == 0 and len(gt) == 0:
        return 0, 0, 0, 0.0
    if len(pred) == 0:
        return 0, 0, int(len(gt)), 0.0
    if len(gt) == 0:
        return 0, int(len(pred)), 0, 0.0
    ious = _iou_matrix(pred, gt)
    # Greedy matching by IoU (no threshold) to compute IoU-quality score
    all_pairs: List[Tuple[int, int, float]] = []
    for i in range(ious.shape[0]):
        for j in range(ious.shape[1]):
            all_pairs.append((i, j, float(ious[i, j])))
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    matched_pred = set()
    matched_gt = set()
    matched_ious: List[float] = []
    for i, j, iou_val in all_pairs:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        matched_ious.append(iou_val)
    # Counts using the provided TP IoU threshold
    tp = int(sum(1 for v in matched_ious if v >= match_iou))
    fp = int(len(pred) - tp)
    fn = int(len(gt) - tp)
    # Mean IoU over GT annotations; unmatched GT contributes 0
    mean_iou_over_gt = float(sum(matched_ious) / max(1, len(gt)))
    return tp, fp, fn, mean_iou_over_gt


@dataclass
class SweepResult:
    conf: float
    iou: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    fp_per_image: float
    mean_iou: float


def _evaluate_combo(
    model: YOLO,
    image_paths: List[Path],
    conf: float,
    iou: float,
    match_iou: float,
    labels_dir: Path,
) -> SweepResult:
    # Run model inference on all images
    paths = [str(p) for p in image_paths]
    results = model.predict(
        paths, conf=float(conf), iou=float(iou), batch=int(DETECTOR_BATCH_SIZE), stream=True, save=False, verbose=False
    )

    total_tp = total_fp = total_fn = 0
    total_mean_iou = 0.0
    num_images = len(image_paths)

    for r in tqdm(results, total=num_images, desc=f'eval conf={conf:.2f} iou={iou:.2f}', leave=False):
        h, w = r.orig_img.shape[:2]

        # Predictions
        if r.boxes is None or len(r.boxes) == 0:
            pred = np.zeros((0, 4), dtype=np.float32)
        else:
            pred = _to_numpy(r.boxes.xyxy).astype(np.float32)

        # Load GT
        img_path = Path(r.path)
        label_path = labels_dir / (img_path.stem + '.txt')
        gt = _load_yolo_labels(label_path, int(w), int(h))

        tp, fp, fn, mean_iou = _match_tp_fp_fn(pred, gt, match_iou)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_mean_iou += mean_iou

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 1.0
    fp_per_image = total_fp / max(1, num_images)
    mean_iou_dataset = total_mean_iou / max(1, num_images)
    return SweepResult(conf, iou, total_tp, total_fp, total_fn, precision, recall, f1, fp_per_image, mean_iou_dataset)


def _to_numpy(tensor_or_array):
    try:
        return tensor_or_array.cpu().numpy()
    except AttributeError:
        return np.array(tensor_or_array)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Bayesian optimization of YOLO detection thresholds on labeled dataset.'
    )
    parser.add_argument('--dataset-dir', type=str, default=str(project_root / 'train/detection/windsurf_dataset'))
    parser.add_argument('--labels-dir', type=str, default='', help='If labels are in a separate dir, set it here')
    parser.add_argument('--model', type=str, default=str(YOLO_MODEL_PATH), help='Path to YOLO .pt model')
    parser.add_argument('--conf-min', type=float, default=0.25)
    parser.add_argument('--conf-max', type=float, default=0.65)
    parser.add_argument('--nms-min', type=float, default=0.50)
    parser.add_argument('--nms-max', type=float, default=0.70)
    parser.add_argument('--match-iou', type=float, default=0.95, help='IoU threshold for TP match vs GT')
    parser.add_argument('--fp-per-image-lower', type=float, default=0.05)
    parser.add_argument('--fp-per-image-upper', type=float, default=0.10)
    parser.add_argument('--trials', type=int, default=60, help='Number of Optuna trials')
    parser.add_argument('--report-json', type=str, default='', help='Optional path to write full sweep results JSON')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        print(f'Dataset not found: {dataset_dir}')
        return
    labels_dir = Path(args.labels_dir).resolve() if args.labels_dir else dataset_dir

    images = _list_images(dataset_dir)[-50:]
    if not images:
        print(f'No images found in {dataset_dir}')
        return

    model_path = Path(args.model)
    if not model_path.exists():
        print(f'Model not found: {model_path}')
        return

    model = YOLO(model=str(model_path), verbose=False)

    results: List[SweepResult] = []
    lower = float(args.fp_per_image_lower)
    upper = float(args.fp_per_image_upper)

    def objective(trial: optuna.trial.Trial) -> float:
        conf = trial.suggest_float('conf', float(args.conf_min), float(args.conf_max))
        nms = trial.suggest_float('nms_iou', float(args.nms_min), float(args.nms_max))
        res = _evaluate_combo(model, images, conf, nms, float(args.match_iou), labels_dir)
        results.append(res)
        # Optimize primarily for mean IoU (box accuracy). Optional penalty for excess FP/image.
        score = float(res.mean_iou)
        if res.fp_per_image > upper:
            score -= 1.0  # strong penalty
        elif res.fp_per_image > lower:
            score -= 0.2 * (res.fp_per_image - lower) / max(1e-6, upper - lower)
        return score

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=max(1, int(args.trials)))

    chosen: SweepResult | None = None
    reason = ''
    # Try lower bound first
    eligible_lower = [r for r in results if r.fp_per_image <= args.fp_per_image_lower]
    if eligible_lower:
        chosen = max(eligible_lower, key=lambda r: (r.mean_iou, -r.fp_per_image))
        reason = f'≤ {args.fp_per_image_lower:.3f}'
    else:
        eligible_upper = [r for r in results if r.fp_per_image <= args.fp_per_image_upper]
        if eligible_upper:
            chosen = max(eligible_upper, key=lambda r: (r.mean_iou, -r.fp_per_image))
            reason = f'≤ {args.fp_per_image_upper:.3f}'
        else:
            # Fall back to global best mean IoU (violates constraint)
            chosen = max(results, key=lambda r: r.mean_iou) if results else None
            reason = 'no pair satisfies FP/image constraint'

    if chosen is None:
        print('No results produced. Check dataset and model.')
        return

    print('Best thresholds:')
    print(f'  conf: {chosen.conf:.3f}, nms_iou: {chosen.iou:.3f}  (constraint {reason})')
    print('Metrics:')
    print(f'  TP: {chosen.tp}, FP: {chosen.fp}, FN: {chosen.fn}, images: {len(images)}')
    print(f'  precision: {chosen.precision:.4f}, recall: {chosen.recall:.4f}, f1: {chosen.f1:.4f}')
    print(f'  mean IoU (per-image avg over GT): {chosen.mean_iou:.4f}')
    print(f'  FP/image: {chosen.fp_per_image:.4f}')

    # Show top-5 by F1 among those within upper constraint
    print('\nTop-5 combos within upper FP/image constraint (by mean IoU):')
    top = sorted(
        [r for r in results if r.fp_per_image <= args.fp_per_image_upper], key=lambda r: (-r.mean_iou, r.fp_per_image)
    )[:5]
    for r in top:
        print(
            f'  conf {r.conf:.3f} | iou {r.iou:.3f} | meanIoU {r.mean_iou:.4f} | f1 {r.f1:.4f} | FP/img {r.fp_per_image:.4f}'
        )

    if args.report_json:
        out = {
            'chosen': chosen.__dict__,
            'all_results': [r.__dict__ for r in results],
            'dataset_dir': str(dataset_dir),
            'model': str(model_path),
            'conf_range': [args.conf_min, args.conf_max],
            'nms_range': [args.nms_min, args.nms_max],
            'match_iou': args.match_iou,
            'constraints': {
                'fp_per_image_lower': args.fp_per_image_lower,
                'fp_per_image_upper': args.fp_per_image_upper,
            },
        }
        with open(args.report_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f'Wrote report to {args.report_json}')


if __name__ == '__main__':
    main()
