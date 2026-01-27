Orientation Classifier Training

End-to-end pipeline to train a YOLOv8 or YOLO11 classification model for detecting frame rotation in 90° increments (0°, 90°, 180°, 270°).

Features
- Automated Dataset Generation: Extracts frames from upright (0°) videos and applies rotations to create a labeled 4-class dataset.
- Orientation-Safe Training: Explicitly disables flips and random rotations (fliplr, flipud, degrees, auto_augment) in the Ultralytics trainer to prevent label corruption.
- Temporary Storage: Generates the dataset in a temporary directory and deletes it automatically after training (override with --keep-dataset).
- Class Balancing: Optional --balance flag ensures equal distribution of rotation classes regardless of frame counts.
- Configurable Sampling: Control dataset size via --sample-prob to process a percentage of video frames.

Usage
python train.py \
  --videos ./path/to/upright_videos \
  --outdir ./runs \
  --sample-prob 0.10 \
  --balance \
  --epochs 30 \
  --batch 64 \
  --imgsz 320

Primary Arguments
- --videos: Folder containing source videos (must be upright/0°).
- --sample-prob: Probability (0.0 to 1.0) of sampling any given frame.
- --weights: Initial model weights (default: yolov8n-cls.pt).
- --balance: Ensures even class distribution during sampling.
- --keep-dataset: Skips the cleanup of generated JPEG images.

Deployment
The inference pipeline typically expects the trained weights at:
video_processing/inference/weights/orientation_fixer/best.pt

TODO
- Add support for --resize-shorter during dataset generation to speed up training on high-resolution source videos.
