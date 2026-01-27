# Windsurfer Detection Training – Quick Guide

## Overview

This guide walks you through the two training loops used in this repo:

1. **Detection (bbox)**: train a YOLO detector to find windsurfers + sails in full frames.
2. **Pose (bbox + 2 keypoints)**: train a YOLO-pose model that predicts:
   - `boom_mast` (boom/mast intersection)
   - `mast_tip`

The **pose model** is what the tracking/player pipeline relies on to compute stable per-frame `anchor` + `scale` for the focused view.

This file also documents the **active learning** workflow used to make keypoint labeling tractable.

---

This guide covers:

- Annotate frames from videos with `annotator.py`
- Review/fix labels with `annotation_editor.py`
- Optionally add negative samples with `negative_sample_creation.py`
- Train the detector with `train.py`

Tip: On Windows, run commands in Git Bash or WSL for best compatibility.

For the full system context (how weights are used in the pipeline / web app), see `documentation/README.md`.

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
Keypoints are expensive to label. The workflow here is designed to spend human time only where the model is uncertain/wrong.

#### Mental model

- **Bboxes** are relatively quick to label and are reused for pose training.
- **Keypoints** (boom/mast + mast tip) are the “expensive” signal; we grow that set gradually.
- Pseudo labels are only used when the model prediction is “safe enough” (gated), so they don’t poison the dataset.

#### What lives where (pose project layout)

Given `--out ./pose_projects/boom_mast_v1`:

- `pose_index.yaml` — which image keys exist and which split they belong to (`train` / `val`).
- `labels_pose/<split>/*.txt` — **manual** pose labels (source of truth).
- `labels_pose_pseudo/<split>/*.txt` — **pseudo** pose labels (accepted by gates).
- `dryruns/run_<timestamp>/...` — analysis output from pseudo-label dry runs (for inspection).

#### Loop (recommended)

1. **Seed**: manually label ~100–300 representative frames (diverse lighting, distances, angles, wave sails if relevant).
2. **Train**: `python train_pose.py ...` to get a first `best.pt`.
3. **Pseudo-label**: run `pseudo_label_pose.py` with strict gates (next section).
4. **Inspect + fix**:
   - open the annotator in `--label-source pseudo` mode,
   - convert “mostly correct” pseudo labels into manual,
   - discard bad ones.
5. **Retrain**: include both manual + pseudo (`--include-pseudo`) and repeat 2–4 rounds.

Stop when additional rounds stop reducing the number of manual fixes needed (diminishing returns).

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

#### Why these gates exist (what they protect against)

- `--iou ...` ensures the model’s bbox matches the *ground-truth* bbox well (so keypoints likely refer to the right object).
- `--kp-conf ...` prevents accepting low-confidence keypoints (common when mast tip is occluded).
- `--require-all-boxes` prevents partial labeling when multiple surfers are present (reduces identity/keypoint swaps).
- `--require-mast-above` enforces a geometric sanity check (mast tip should be above boom/mast in image space for typical beach-shot footage).

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

Checkpoint saving (every 50 epochs by default):
- Detection trainer: `python train.py ... --save-period 50`
- Pose trainer (full-frame): `python train_pose.py ... --save-period 50`
- Pose trainer (crops): `python train/pose/train.py ... --save-period 50`

Compare multiple pose checkpoints (renders only predicted crops in a grid):
```bash
python compare_pose_checkpoints_grid.py --run train/detection/runs/pose_seed200 --epochs 50 100 150 200 --images ./windsurf_dataset --out ./tmp/checkpoint_grids
python compare_pose_checkpoints_grid.py --run train/detection/runs/pose_seed200 --epochs 50 100 150 200 --video ../../tmp/ingress/some_video.mp4 --out ./tmp/checkpoint_grids
```

Sanity-check that pose labels reuse the original bbox labels (confirms pseudo labeling is not writing predicted bboxes):
```bash
python check_pose_label_consistency.py --src ./windsurf_dataset --pose ./pose_projects/boom_mast_v1 --label-source both
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
