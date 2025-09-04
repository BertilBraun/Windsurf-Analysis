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


