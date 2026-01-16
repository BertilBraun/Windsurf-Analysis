#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from ultralytics import YOLO


KP_NAMES = ["boom_mast", "mast_tip"]
SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Active-learning helper: run a trained YOLO-pose model over the dataset and write pose labels for samples\n"
            "whose predicted boxes match the existing GT boxes well (IoU gate) and whose keypoints pass simple checks.\n"
            "Default behavior is a DRYRUN that writes to a separate pose-project folder you can inspect with view_pose_labels.py.\n"
            "To persist pseudo labels into the pose project, use --mode write (writes to labels_pose_pseudo/...)."
        )
    )
    p.add_argument("--src", type=Path, required=True, help="Detection dataset root (images + bbox labels).")
    p.add_argument("--pose", type=Path, required=True, help="Pose project directory created by annotator.")
    p.add_argument("--model", type=Path, required=True, help="Trained pose model weights (e.g. .../weights/best.pt).")
    p.add_argument(
        "--mode",
        choices=["dryrun", "write"],
        default="dryrun",
        help="dryrun: write labels into a new dryrun pose-project folder; write: write into labels_pose_pseudo/ in --pose",
    )
    p.add_argument(
        "--dryrun-out",
        type=Path,
        default=None,
        help="Optional explicit output pose-project folder for dryrun mode (default: <pose>/dryruns/<timestamp>).",
    )
    p.add_argument(
        "--write-subdir",
        type=str,
        default="labels_pose_pseudo",
        help="Subdir under --pose to store pseudo labels when --mode write (default: labels_pose_pseudo).",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold for predictions.")
    p.add_argument("--iou", type=float, default=0.75, help="Min IoU between predicted box and GT box to accept.")
    p.add_argument("--kp-conf", type=float, default=0.30, help="Min keypoint confidence (if available) to accept.")
    p.add_argument("--require-mast-above", action="store_true", help="Require mast_tip to be above boom_mast (y smaller).")
    p.add_argument(
        "--bbox-margin",
        type=float,
        default=0.05,
        help="Keypoints must lie within GT bbox expanded by this margin (fraction of bbox size).",
    )
    p.add_argument(
        "--require-all-boxes",
        action="store_true",
        help="Only write a label if ALL GT boxes in the image are confidently pseudo-labeled.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing pseudo labels (does not touch manual).")
    p.add_argument("--max-images", type=int, default=0, help="Limit number of images processed (0 = no limit).")
    p.add_argument(
        "--predict-batch",
        type=int,
        default=8,
        help="Inference batch size in number of images (keeps memory bounded; lower this if you hit OOM).",
    )
    p.add_argument("--device", type=str, default="auto")
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


def _read_bboxes(label_path: Path) -> list[YoloBBox]:
    if not label_path.exists():
        return []
    out: list[YoloBBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
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


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def _kp_in_expanded_bbox(
    kp: tuple[float, float],
    bbox: tuple[float, float, float, float],
    *,
    margin: float,
) -> bool:
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = bw * float(margin)
    my = bh * float(margin)
    ex1 = x1 - mx
    ey1 = y1 - my
    ex2 = x2 + mx
    ey2 = y2 + my
    x, y = kp
    return ex1 <= x <= ex2 and ey1 <= y <= ey2


def _pose_label_path(pose_dir: Path, *, split: str, key: str) -> Path:
    return pose_dir / "labels_pose" / split / f"{key}.txt"

def _pseudo_label_path(pose_dir: Path, *, split: str, key: str, subdir: str) -> Path:
    return pose_dir / subdir / split / f"{key}.txt"


def _write_pose_label(
    out_path: Path,
    *,
    gt_bboxes: list[YoloBBox],
    kps_by_box: list[list[tuple[float, float, int]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, b in enumerate(gt_bboxes):
        parts = [
            str(int(b.cls_id)),
            f"{float(b.cx):.6f}",
            f"{float(b.cy):.6f}",
            f"{float(b.w):.6f}",
            f"{float(b.h):.6f}",
        ]
        for (x, y, v) in kps_by_box[i][:2]:
            vv = 1 if int(v) > 0 else 0
            xx = float(x) if vv > 0 else 0.0
            yy = float(y) if vv > 0 else 0.0
            parts.extend([f"{xx:.6f}", f"{yy:.6f}", str(vv)])
        lines.append(" ".join(parts))
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _iter_batches(items: list[dict], batch_size: int) -> list[list[dict]]:
    batch_size = max(1, int(batch_size))
    out: list[list[dict]] = []
    for i in range(0, len(items), batch_size):
        out.append(items[i : i + batch_size])
    return out


def main() -> int:
    args = _parse_args()
    src_dir = Path(args.src)
    pose_dir = Path(args.pose)
    if not src_dir.exists():
        raise SystemExit(f"--src does not exist: {src_dir}")
    if not pose_dir.exists():
        raise SystemExit(f"--pose does not exist: {pose_dir}")
    if not Path(args.model).exists():
        raise SystemExit(f"--model does not exist: {args.model}")

    index = _load_index(pose_dir)
    items = index.get("items", [])
    assert isinstance(items, list)

    mode = str(args.mode)
    write_subdir = str(args.write_subdir)

    out_pose_dir: Path
    if mode == "dryrun":
        if args.dryrun_out is not None:
            out_pose_dir = Path(args.dryrun_out)
        else:
            import datetime

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_pose_dir = pose_dir / "dryruns" / f"run_{ts}"
        out_pose_dir.mkdir(parents=True, exist_ok=True)
        # Copy pose_index.yaml so the viewer can navigate consistently.
        (out_pose_dir / "pose_index.yaml").write_text(_index_path(pose_dir).read_text(encoding="utf-8"), encoding="utf-8")
        for sp in ("train", "val"):
            (out_pose_dir / "labels_pose" / sp).mkdir(parents=True, exist_ok=True)
    elif mode == "write":
        out_pose_dir = pose_dir
        for sp in ("train", "val"):
            (out_pose_dir / write_subdir / sp).mkdir(parents=True, exist_ok=True)
    else:
        raise SystemExit(f"Unknown --mode: {mode}")

    # Build list of candidates (only those with GT bbox labels, and without pseudo labels unless --overwrite)
    candidates: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        key = str(it.get("key", ""))
        split = str(it.get("split", "train"))
        rel = str(it.get("src_rel", ""))
        if not key or split not in ("train", "val") or not rel:
            continue
        img_path = src_dir / Path(rel)
        if not img_path.exists() or img_path.suffix.lower() not in SUPPORTED_IMG_EXTS:
            continue
        gt_label = img_path.with_suffix(".txt")
        if not gt_label.exists():
            continue
        if not _read_bboxes(gt_label):
            continue
        if mode == "dryrun":
            out_pose = _pose_label_path(out_pose_dir, split=split, key=key)
        else:
            out_pose = _pseudo_label_path(out_pose_dir, split=split, key=key, subdir=write_subdir)

        if out_pose.exists() and not bool(args.overwrite):
            continue
        candidates.append({"key": key, "split": split, "img_path": img_path})

    if not candidates:
        raise SystemExit("No pseudo-label candidates found (maybe everything is already labeled?).")

    max_images = int(args.max_images)
    if max_images > 0:
        candidates = candidates[:max_images]

    model = YOLO(str(args.model))

    counts = {
        "images_seen": 0,
        "images_written": 0,
        "images_skipped": 0,
        "boxes_total": 0,
        "boxes_accepted": 0,
        "mode": mode,
        "out_pose_dir": str(out_pose_dir),
    }

    # Predict in small batches to avoid large allocations when passing a huge list of sources.
    for batch in _iter_batches(candidates, int(args.predict_batch)):
        batch_by_path: dict[str, dict] = {str(Path(c["img_path"]).resolve()): c for c in batch}
        batch_sources = [str(c["img_path"]) for c in batch]

        results = model.predict(
            source=batch_sources,
            conf=float(args.conf),
            device=str(args.device),
            verbose=False,
            save=False,
            save_txt=False,
        )

        for res in results:
            img_path = Path(getattr(res, "path", "")).resolve()
            c = batch_by_path.get(str(img_path))
            if c is None:
                continue
            key = str(c["key"])
            split = str(c["split"])

            orig_shape = getattr(res, "orig_shape", None)
            if not (isinstance(orig_shape, (list, tuple)) and len(orig_shape) >= 2):
                counts["images_skipped"] += 1
                continue
            img_h, img_w = int(orig_shape[0]), int(orig_shape[1])
            if img_h <= 0 or img_w <= 0:
                counts["images_skipped"] += 1
                continue

            gt_bboxes = _read_bboxes(img_path.with_suffix(".txt"))
            if not gt_bboxes:
                counts["images_skipped"] += 1
                continue
            gt_xyxy = [b.to_xyxy_abs(img_w=img_w, img_h=img_h) for b in gt_bboxes]
            counts["images_seen"] += 1
            counts["boxes_total"] += len(gt_bboxes)

            boxes = getattr(res, "boxes", None)
            kps = getattr(res, "keypoints", None)
            if boxes is None or kps is None:
                counts["images_skipped"] += 1
                continue

            xyxy_t = getattr(boxes, "xyxy", None)
            conf_t = getattr(boxes, "conf", None)
            cls_t = getattr(boxes, "cls", None)
            kxy_t = getattr(kps, "xy", None)
            kconf_t = getattr(kps, "conf", None)
            if xyxy_t is None or conf_t is None or cls_t is None or kxy_t is None:
                counts["images_skipped"] += 1
                continue

            try:
                pred_xyxy = xyxy_t.cpu().numpy().tolist()
                pred_conf = conf_t.cpu().numpy().tolist()
                pred_cls = cls_t.cpu().numpy().tolist()
                pred_kxy = kxy_t.cpu().numpy().tolist()
                pred_kconf = kconf_t.cpu().numpy().tolist() if kconf_t is not None else None
            except Exception:
                counts["images_skipped"] += 1
                continue

            # Greedy match GT boxes to predictions by IoU (one prediction per GT)
            used_preds: set[int] = set()
            accepted_flags = [False] * len(gt_bboxes)
            out_kps: list[list[tuple[float, float, int]]] = [[(0.0, 0.0, 0), (0.0, 0.0, 0)] for _ in gt_bboxes]

            for gi, gbox in enumerate(gt_xyxy):
                best_pi = None
                best_iou = 0.0
                for pi, pbox in enumerate(pred_xyxy):
                    if pi in used_preds:
                        continue
                    iou_val = _iou(gbox, (float(pbox[0]), float(pbox[1]), float(pbox[2]), float(pbox[3])))
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_pi = pi
                if best_pi is None:
                    continue

                if best_iou < float(args.iou):
                    continue
                if float(pred_conf[best_pi]) < float(args.conf):
                    continue

                # Extract keypoints 0..1, normalize, and run checks.
                kxy = pred_kxy[best_pi]
                if not (isinstance(kxy, list) and len(kxy) >= 2):
                    continue

                kp_out: list[tuple[float, float, int]] = []
                ok = True
                for kpi in range(2):
                    try:
                        px = float(kxy[kpi][0])
                        py = float(kxy[kpi][1])
                    except Exception:
                        ok = False
                        break
                    nx = max(0.0, min(1.0, px / float(img_w)))
                    ny = max(0.0, min(1.0, py / float(img_h)))
                    v = 1
                    if pred_kconf is not None:
                        kc = _as_float(pred_kconf[best_pi][kpi])
                        if not (kc == kc) or kc < float(args.kp_conf):
                            v = 0
                            nx, ny = 0.0, 0.0
                    if v > 0:
                        if not _kp_in_expanded_bbox((px, py), gbox, margin=float(args.bbox_margin)):
                            ok = False
                            break
                    kp_out.append((nx, ny, v))

                if not ok:
                    continue

                if bool(args.require_mast_above):
                    boom = kp_out[0]
                    tip = kp_out[1]
                    if boom[2] > 0 and tip[2] > 0 and not (tip[1] < boom[1]):
                        continue

                # Accept if both keypoints are visible (v=1)
                if not (kp_out[0][2] > 0 and kp_out[1][2] > 0):
                    continue

                used_preds.add(best_pi)
                accepted_flags[gi] = True
                out_kps[gi] = kp_out

            if bool(args.require_all_boxes) and not all(accepted_flags):
                counts["images_skipped"] += 1
                continue

            if not any(accepted_flags):
                counts["images_skipped"] += 1
                continue

            counts["boxes_accepted"] += sum(1 for f in accepted_flags if f)
            if mode == "dryrun":
                out_path = _pose_label_path(out_pose_dir, split=split, key=key)
            else:
                out_path = _pseudo_label_path(out_pose_dir, split=split, key=key, subdir=write_subdir)
            counts["images_written"] += 1
            _write_pose_label(out_path, gt_bboxes=gt_bboxes, kps_by_box=out_kps)

    print(yaml.safe_dump(counts, sort_keys=False).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
