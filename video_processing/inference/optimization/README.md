Tools for creating ground truth "golden" datasets and optimizing tracking hyperparameters using Bayesian optimization (Optuna) and interactive GUI tuners.

### Annotation & Evaluation
*   **annotate_tracklets.py**: GUI tool to manually merge and label tracklets into a "golden" ground truth standard. Supports rapid assignment, discarding, and undoing via keyboard shortcuts.
*   **compare_trackers.py**: Benchmarks multiple tracking algorithms (e.g., BoT-SORT, Discrete ILP, OC-SORT) against golden data using pairwise F1 scores and execution time.
*   **evaluate_tracker.py**: Runs a specific tracking pipeline and calculates clustering metrics (Precision, Recall, F1, Rand Index) against ground truth.

### Automated Optimization (Optuna)
*   **optimize_tracker.py**: Unified script to optimize hyperparameters for the Preprocessor, Discrete ILP, and Iterative ILP trackers using a shared worker pool for parallel evaluation.
*   **optimize_detection_thresholds.py**: Finds optimal YOLO confidence and NMS thresholds by maximizing mean IoU on a labeled image dataset while respecting false-positive-per-image constraints.
*   **optimize_kalman.py**: Minimizes Kalman Filter prediction error (1 - IoU) by tuning process and measurement noise weights against golden tracks.
*   **optimize_pairwise_association.py**: Tunes hyperparameters for motion (Mahalanobis distance) and embedding (Cosine/Chi-square) association costs to maximize separation between positive and negative pairs.
*   **optimize_discrete_tracker.py**: Specific Bayesian optimization script for the Discrete ILP tracker's linking costs and window radii.

### Interactive Tuning GUIs
*   **tune_detection_thresholds.py**: Visual tool to adjust YOLO confidence, NMS, and containment suppression thresholds in real-time on a video stream. Supports model switching and class filtering.
*   **tune_greedy_preprocessor.py**: Interactive overlay to tune greedy tracklet merging parameters (IoU, Cosine Similarity, Max Gap, and EMA alpha) with live track visualization.

### Utilities
*   **optimization_util.py**: Shared logic for loading "golden" pickle files with backward compatibility, calculating pairwise clustering scores, and managing multiprocessing worker pools for optimization trials.

### TODO
*   Standardize the "golden" metadata format across all scripts to ensure 100% compatibility between older annotation files and newer optimization routines.
*   Re-enable and verify post-processing steps in compare_trackers.py.
