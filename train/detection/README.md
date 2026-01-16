# Windsurfer Detection Training – Quick Guide

## Overview
This guide walks you through creating a high‑quality detection dataset and training the YOLOv11 model:
- Annotate frames from videos with `annotator.py`
- Review/fix labels with `annotation_editor.py`
- Optionally add negative samples with `negative_sample_creation.py`
- Train the detector with `train.py`

Tip: On Windows, run commands in Git Bash or WSL for best compatibility.

## 0) Setup
```bash
git clone https://github.com/BertilBraun/Windsurf-Analysis.git
cd Windsurf-Analysis
pip install -r requirements.txt
cd train/detection
```

## 1) Annotate samples from videos (required)
Samples are taken from your videos and saved as `.jpg` + `.txt` (YOLO format).
```bash
python annotator.py <video_dir> <output_dir> --samples 1000
# example
python annotator.py ../../tmp/ingress "./windsurf_dataset" --samples 1500
```

Controls (essentials):
- Draw box: LMB drag
- Accept/save: Space
- Undo last box: r
- Fine‑tune last box: w/a/s/d/W/A/S/D
- Previous/next frame: , / .
- Empty frame (no boxes): e
- Skip frame (don't save): x

Annotation quality guidelines (critical):
- **Be extremely consistent and tight.** Boxes should closely hug the object.
- **Include the windsurfer and the sail; exclude the board.** The board is often hard to see and inconsistent. Excluding it yields tighter, cleaner labels and better model boxes.
- **Prefer consistency over occasional completeness.** If a feature is only sometimes visible (e.g., board), do not expand boxes to include it.
- **Avoid tiny/ambiguous boxes.** Skip frames that don’t show a clear target.

Output: `output_dir` will contain pairs like `.../image_0001.jpg` and `.../image_0001.txt` with class `0`.

## 2) Review and fix annotations (recommended)
Open images and their YOLO labels to adjust boxes.
```bash
python annotation_editor.py <images_dir>
# example
python annotation_editor.py ./windsurf_dataset
```
Controls (essentials):
- Select box: click inside box
- Draw new box: LMB drag
- Save labels + next image: Space
- Delete selected/last box: r
- Prev/next image: , / .

## 3) Add negative samples (optional)
Create crops with no objects; each saved crop gets an empty `.txt` label.
```bash
python negative_sample_creation.py <images_dir> --output-dir <neg_dir> --min-side 640
# example
python negative_sample_creation.py ./windsurf_dataset --output-dir ./windsurf_dataset/negatives --min-side 640
```
Notes:
- Draw rectangles to define background regions; they are expanded/padded to at least `min-side` in both dimensions.
- Empty labels help the model learn true background.

## 4) Train the model
Point training at the directory that contains your `.jpg`/`.txt` pairs (including negatives, if any).
```bash
python train.py \
  --src ./windsurf_dataset \
  --dst ./datasets/windsurfers \
  --val-ratio 0.05 \
  --epochs 100 \
  --imgsz 640 \
  --batch 0.7 \
  --device auto \
  --base-model yolo11s.pt
```
Details:
- The script prepares the Ultralytics dataset structure, sanitizes labels, writes a YAML, and launches training.
- After training completes, the temporary dataset folder at `--dst` is removed. Training results are written by Ultralytics under `runs/detect/train*`.

## 5) Pose (2-keypoint) annotation + training (optional)
This trains a single YOLO-pose model that predicts both bbox + 2 keypoints per detection:
1) `boom_mast` (boom-mast intersection)
2) `mast_tip`

### 5.1) Annotate keypoints (full frames, multi-box)
This reads your existing bbox labels from `--src` and stores pose labels separately (only for images you annotate).

```bash
python annotator_keypoints_fullframe.py --src ./windsurf_dataset --out ./pose_projects/boom_mast_v1
# to review already-annotated samples too:
python annotator_keypoints_fullframe.py --src ./windsurf_dataset --out ./pose_projects/boom_mast_v1 --show-annotated
```

Review pseudo labels (written via `pseudo_label_pose.py --mode write`) and convert/fix them into manual labels:
```bash
python annotator_keypoints_fullframe.py --src ./windsurf_dataset --out ./pose_projects/boom_mast_v1 --label-source pseudo --only-labeled --write-target manual
```

Quick viewer (bbox + keypoints overlay on the original frame):
```bash
python view_pose_labels.py --src ./windsurf_dataset --pose ./pose_projects/boom_mast_v1 --split val --only-labeled
```

### 5.2) Train pose model
```bash
python train_pose.py --src ./windsurf_dataset --pose ./pose_projects/boom_mast_v1 --base-model yolo11n-pose.pt --device auto
```

### 5.3) Active learning (pseudo-label the remaining samples)
Workflow:
1) Manually label a small seed set (e.g. ~200) with `annotator_keypoints_fullframe.py`.
2) Train a first pose model with `train_pose.py`.
3) Run pseudo-labeling to auto-accept keypoints on the rest where predictions match the GT bboxes well.
4) Inspect/correct with `annotator_keypoints_fullframe.py`, then retrain.

Pseudo-labeling writes pose labels for samples that pass gates (IoU vs GT bbox + keypoint checks):
```bash
python pseudo_label_pose.py ^
  --src ./windsurf_dataset ^
  --pose ./pose_projects/boom_mast_v1 ^
  --model train/detection/runs/pose/weights/best.pt ^
  --iou 0.75 ^
  --conf 0.25 ^
  --kp-conf 0.30 ^
  --require-all-boxes ^
  --require-mast-above
```

Inspect a dryrun output with the viewer:
```bash
python pseudo_label_pose.py --src ./windsurf_dataset --pose ./pose_projects/boom_mast_v1 --model <best.pt>
python view_pose_labels.py --src ./windsurf_dataset --pose ./pose_projects/boom_mast_v1/dryruns/run_<timestamp> --split train --only-labeled
```

Persist pseudo labels into the pose project (stored separately as `labels_pose_pseudo/`):
```bash
python pseudo_label_pose.py --src ./windsurf_dataset --pose ./pose_projects/boom_mast_v1 --model <best.pt> --mode write
python view_pose_labels.py --src ./windsurf_dataset --pose ./pose_projects/boom_mast_v1 --split train --only-labeled --label-source pseudo
```

Train including pseudo labels (copied into the temp dataset with filename prefix `pseudo_`):
```bash
python train_pose.py --src ./windsurf_dataset --pose ./pose_projects/boom_mast_v1 --include-pseudo --pseudo-frac 0.5 --seed 0
```

## Quickstart
You can copy and run the helper script from this folder:
```bash
bash quickstart_train.sh
```

## Dataset quality is everything
- **Consistency > everything.** Use the same inclusion rules everywhere (windsurfer + sail, no board).
- **Tight boxes.** Avoid slack; tight annotations produce tight, usable predictions.
- **Clean negatives.** Include representative backgrounds with empty labels.

With a clean, consistent dataset, the final model yields precise and reliable bounding boxes.
