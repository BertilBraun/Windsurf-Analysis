Tools for benchmarking, evaluating, and optimizing video stabilization algorithms.

### Core Scripts

*   **evaluate_stabilization_methods.py**: Compares stabilization methods (`vidstab`, `gmc`, `masked_vidstab`) against ground truth.
    *   Calculates RMSE and MAE for translation (dx, dy) and rotation (da).
    *   Supports YOLO-based masking to exclude moving objects from motion estimation.
    *   Generates a JSON report of performance metrics.
*   **generate_jittered_video.py**: Creates synthetic "shaky" videos for testing.
    *   Applies random per-frame translations and rotations.
    *   Crops output to remove black borders.
    *   Saves ground truth transforms to a JSON file for use with evaluation scripts.
*   **vidstab_comparison.py**: Benchmarks different keypoint detection methods (GFTT, BRISK, FAST, etc.) within the `vidstab` framework.
    *   Includes `BBoxMaskedVidStab`, a custom estimator that uses object detection tracks to mask out moving foreground objects.
    *   Supports grid testing of smoothing windows and stabilization strength (alpha).
*   **vidstab_gftt_tournament.py**: A hyperparameter optimization tool for GFTT-based stabilization.
    *   Runs a grid search over parameters like corner count, quality level, and processing resolution.
    *   Provides an interactive side-by-side "tournament" viewer to manually pick the best-looking results.
    *   Exports a ranked list of configurations based on user preference.

### Key Features

*   **Masked Motion Estimation**: Ability to ignore specific bounding boxes (from YOLO or `.tracks.pkl`) to ensure stabilization is based on the static background rather than moving subjects.
*   **Synthetic Benchmarking**: End-to-end pipeline for generating known jitter and measuring how accurately different algorithms can recover the original camera path.
*   **Interactive Optimization**: Visual comparison tools to tune stabilization parameters when objective metrics are insufficient.
*   **Tournament UI**: Keyboard-driven interface (1/2 to pick, space to pause, 'r' to restart) for blind comparison of stabilization outputs.

### TODO
*   Integrate more advanced Global Motion Compensation (GMC) variants.
*   Add support for automated "no-reference" video quality metrics (e.g., cropping ratio, distortion).
