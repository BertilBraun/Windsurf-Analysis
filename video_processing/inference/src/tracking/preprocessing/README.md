This folder contains utilities for cleaning and initial stitching of tracks before they are passed to the main tracking or post-processing logic.

### Components

*   **TrackPreProcessor**: The primary entry point that orchestrates the preprocessing pipeline.
*   **GreedyTrackStitcher**: A greedy association engine that merges single-detection tracks into longer segments.
    *   **Motion Scoring**: Uses Kalman Filter state predictions and Camera Motion Compensation (CMC) to calculate Mahalanobis distance.
    *   **Appearance Scoring**: Uses embedding similarity (histograms/feature vectors) with Exponential Moving Average (EMA) updates for track templates.
    *   **Logic**: Categorizes potential matches as `MATCH` (strict), `MAY_MATCH` (loose), or `NO_MATCH` based on configurable probability thresholds.
    *   **Conflict Resolution**: Automatically fades tracks or creates new ones when assignments are ambiguous or overlapping.
*   **FilterNonSurfers**: A heuristic filter to remove spurious tracks.
    *   Identifies "short" tracks (fewer than `min_frames`).
    *   Removes short tracks if their average embedding similarity to established "long" tracks is below a threshold.

### Debugging & Visualization

*   **Heatmaps**: `GreedyTrackStitcher` can generate heatmaps showing motion and appearance probability matrices for active tracks vs. new detections.
*   **Overlays**: Supports Kalman Filter state and detection bounding box overlays for visual verification of stitching decisions.

### TODO
*   Re-enable and validate `FilterNonSurfers` integration within `TrackPreProcessor` (currently commented out).
*   Remove temporary debug visualization code from `GreedyTrackStitcher`.
*   Address the "Isolation logic" placeholder in `_compare_detection_to_track`.
*   Reconcile `FilterNonSurfers` docstring (mentions inflated-bbox overlap) with implementation (uses embedding similarity).
