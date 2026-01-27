Utility scripts for running the video processing pipeline locally, generating visualizations, and optimizing stabilization parameters.

### Core Scripts

*   **local_modal_pipeline_player.py**: A local implementation of the cloud-based Modal pipeline.
    *   Executes the full sequence: orientation correction, YOLO detection, camera stabilization, embedding extraction, and ILP tracking.
    *   Supports multiple stabilizers: `masked_vidstab`, `gmc`, and `vidstab`.
    *   Outputs metadata (`.tracks.pkl`) and transformation JSONs compatible with the project's players.
    *   Optionally launches the local Qt player immediately after processing.
*   **render_homepage_demo.py**: Generates a 3-panel demonstration video for a specific track ID.
    *   **Left Top**: Raw/shaky video.
    *   **Left Bottom**: Tracking overlays with a fading motion trail.
    *   **Right**: A detailed, centered, and zoomed-in view of the rider based on pose-derived anchors.
    *   Can reuse existing pipeline outputs or run a fresh analysis pass.

### Subfolders

*   **stabilization_optimization**: Specialized tools for benchmarking, evaluating, and hyperparameter tuning of stabilization algorithms against ground truth or synthetic jitter.

### TODO

*   Standardize CLI arguments between the local pipeline and the demo renderer.
*   Add support for batch processing multiple videos in `local_modal_pipeline_player.py`.
