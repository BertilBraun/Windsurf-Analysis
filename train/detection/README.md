Tools for annotating and training YOLO-based detection and pose estimation models.

### Detection Annotation
*   **annotator.py**: Extracts random frames from videos for manual bounding box labeling.
*   **annotation_editor.py**: Review and edit existing YOLO format bounding boxes on images.
*   **negative_sample_creation.py**: Extracts background crops (minimum 640x640) and generates empty label files to reduce false positives.

### Pose & Keypoint Annotation
*   **annotator_keypoints_fullframe.py**: Precision labeling for `boom_mast` and `mast_tip` keypoints using zoomed-in bounding box crops.
*   **pseudo_label_pose.py**: Active learning helper that uses a trained model to generate pose labels, gated by IoU and confidence.
*   **view_pose_labels.py**: Visualizer for pose projects; renders bboxes and keypoints (manual or pseudo) over full-frame images.

### Training & Evaluation
*   **train.py**: End-to-end detection pipeline: restructures data, sanitizes labels (clamping/filtering tiny boxes), and fine-tunes YOLOv11.
*   **train_pose.py**: Trains YOLO-pose models; supports mixing manual labels with a configurable fraction of pseudo-labels.
*   **compare_pose_checkpoints_grid.py**: Generates a visual grid comparing inference results from multiple model checkpoints or epochs.

### Shared Utilities
*   **screen_utils.py**: Detects screen resolution and overlays UI warnings if images exceed display bounds.

### Common Controls
*   **LMB Drag**: Draw new box.
*   **LMB Click**: Select existing box.
*   **Space**: Save and advance to next sample.
*   **r**: Delete selected or last box.
*   **, / .**: Navigate previous/next image.
*   **w/a/s/d**: Move or grow box edges.
*   **W/A/S/D**: Shrink box edges.
*   **Esc**: Quit.

### TODO
*   Implement `check_pose_label_consistency.py` for dataset validation.
