Core execution engine and hyperparameter optimization suite for the windsurfing video processing pipeline.

### Subfolders

*   **[src](./src)**: Core pipeline source code.
    *   **Orchestration**: End-to-end processing from raw video to stabilized, annotated highlights.
    *   **Tracking**: Implementations for ILP (Discrete/Iterative), Greedy, and OC-SORT trackers.
    *   **Motion**: Global Camera Motion (GMC) estimation and Kalman Filter motion compensation.
    *   **Interfaces**: CLI for batch processing and a PySide6-based interactive video player.
*   **[optimization](./optimization)**: Performance tuning and evaluation tools.
    *   **Annotation**: GUI for creating "golden" ground truth datasets from tracklets.
    *   **Automated Tuning**: Optuna-based Bayesian optimization for Kalman filters, detection thresholds, and association costs.
    *   **Benchmarking**: Scripts to compare tracker accuracy (F1, IoU, Rand Index) against ground truth.
    *   **Interactive Tuning**: Real-time GUIs to visually adjust detection and merging parameters.

### Key Capabilities

*   **Automated Pipeline**: Handles detection, ReID embedding extraction, stabilization, and tracking in a unified workflow.
*   **Orientation Correction**: Automatically detects and fixes video rotation using YOLO classification.
*   **Hyperparameter Optimization**: Maximizes tracking metrics by parallelizing evaluation across shared worker pools.
*   **Visualization**: Generates annotated videos, individual track crops, and metadata for interactive review.

### TODO

*   Standardize "golden" metadata formats to ensure 100% compatibility between annotation tools and optimization routines.
*   Optimize GMC feature tracking performance for 4K video inputs.
*   Standardize the `Tracker` protocol usage across all implementations in the main orchestrator.
*   Re-verify post-processing steps in the tracker comparison suite.
