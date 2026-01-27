Training pipelines and annotation tools for computer vision models.

### Subfolders

*   **detection/**: YOLO-based detection and pose estimation.
    *   **Annotation**: Manual bounding box and keypoint labeling, pseudo-labeling for active learning, and negative sample generation.
    *   **Training**: End-to-end pipelines for YOLOv11 detection and pose models, including label sanitation and checkpoint comparison.
    *   **Utilities**: Visualizers for pose labels and screen-aware UI helpers.
*   **rotation-classification/**: Orientation detection (0°, 90°, 180°, 270°).
    *   **Dataset Generation**: Automated creation of 4-class datasets from upright source videos.
    *   **Training**: Specialized YOLO classification training that disables orientation-conflicting augmentations (flips/rotations).

### TODO
*   Implement `check_pose_label_consistency.py` for dataset validation.
*   Add support for `--resize-shorter` during rotation dataset generation to improve training speed.
