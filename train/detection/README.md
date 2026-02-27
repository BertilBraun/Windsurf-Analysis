Tools for annotating and training YOLO-based detection and pose estimation models.

### Detection Annotation
*   **annotator.py**: Extracts random frames from videos for manual bounding box labeling.
*   **annotation_editor.py**: Review and edit existing YOLO format bounding boxes on images.
*   **negative_sample_creation.py**: Extracts background crops (minimum 640x640) and generates empty label files to reduce false positives.
*   **pseudo_label_bbox.py**: Generates conservative pseudo-labeled bbox samples from videos into a review folder.
*   **promote_pseudo_bbox_samples.py**: Promotes reviewed pseudo-labeled bbox samples into `windsurf_dataset` with collision-safe naming.

### Pose & Keypoint Annotation
*   **annotator_keypoints_fullframe.py**: Precision labeling for `boom_mast` and `mast_tip` keypoints using zoomed-in bounding box crops (use `--sync-index` to pick up newly added detection samples).
*   **pseudo_label_pose.py**: Active learning helper that uses a trained model to generate pose labels, gated by IoU and confidence.
*   **view_pose_labels.py**: Visualizer for pose projects; renders bboxes and keypoints (manual or pseudo) over full-frame images.

### Training & Evaluation
*   **train.py**: End-to-end detection pipeline: restructures data, sanitizes labels (clamping/filtering tiny boxes), and fine-tunes YOLOv11.
*   **train_pose.py**: Trains YOLO-pose models; supports mixing manual labels with a configurable fraction of pseudo-labels.
*   **compare_pose_checkpoints_grid.py**: Generates a visual grid comparing inference results from multiple model checkpoints or epochs.

### Shared Utilities
*   **screen_utils.py**: Detects screen resolution and overlays UI warnings if images exceed display bounds.

### BBox Pseudo-Labeling Workflow
1. Generate pseudo bbox labels from new videos (review-first):
   `python train/detection/pseudo_label_bbox.py --videos "C:\Users\berti\Downloads\training videos from kasper" --model "train/detection/runs/detect/train8/weights/best.pt" --out "train/detection/neg/pseudo_bbox_review"`
2. Review and edit pseudo labels:
   `python train/detection/annotation_editor.py "train/detection/neg/pseudo_bbox_review"`
3. Promote reviewed samples into the main bbox dataset:
   `python train/detection/promote_pseudo_bbox_samples.py --src "train/detection/neg/pseudo_bbox_review" --dst "train/detection/windsurf_dataset" --copy`
4. Continue training flow:
   - Detection: `python train/detection/train.py --src train/detection/windsurf_dataset --base-model yolo11m.pt`
   - Sync keypoint index: `python train/detection/annotator_keypoints_fullframe.py --src train/detection/windsurf_dataset --out train/detection/pose_projects/boom_mast_v1 --sync-index`

Notes:
*   Manual bbox annotation with `annotator.py` is still the fallback when pseudo labels are weak.
*   Conservative defaults are tuned for lower cleanup effort; relax `--conf`, `--min-box-side`, `--edge-margin-frac`, or increase `--max-boxes-per-image` to accept more candidates.

### Common Controls
*   **LMB Drag**: Draw new box.
*   **Q / E**: Rotate 90° CCW / CW (annotator only).
*   **LMB Click**: Select existing box.
*   **Space**: Save and advance to next sample.
*   **r**: Delete selected or last box.
*   **, / .**: Navigate previous/next image.
*   **w/a/s/d**: Move or grow box edges.
*   **W/A/S/D**: Shrink box edges.
*   **Esc**: Quit.

### TODO
*   Implement `check_pose_label_consistency.py` for dataset validation.
