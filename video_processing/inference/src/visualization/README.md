Tools for rendering tracking results, generating debug visualizations, and processing video stabilization.

### Core Components

*   **annotation_drawer.py**: Standard OpenCV-based drawing for real-time or post-process visualization.
    *   Draws bounding boxes with unique, track-persistent colors.
    *   Renders "tracking trails" showing the historical path of an object.
    *   Handles automatic label background coloring and text contrast adjustment.
*   **debug_drawer.py**: High-level framework for complex debug video layouts.
    *   **DebugCanvas**: Manages a global image with collision-aware label placement to prevent text overlap.
    *   **DebugView**: Provides a coordinate-transformed sub-region (e.g., for side-by-side frame comparisons).
    *   Visualizes similarity metrics (IoU, Cosine similarity, distance) between detections across frames.
*   **stabilize.py**: Camera motion estimation and video stabilization.
    *   Supports multiple estimation backends: `VidStab`, `GMC` (Global Motion Compensation), and a custom `MaskedVidStabEstimator`.
    *   Provides keypoint masking to exclude moving objects (bounding boxes) from motion estimation.
    *   Computes absolute correction transforms for direct rendering or playback.
*   **track_graph_viz.py**: High-level visualization of track lifespans and merge logic.
    *   **Timeline Style**: Gantt-style chart using Matplotlib showing track duration and potential merge edges.
    *   **Graph Style**: Graphviz-based node-edge diagram for inspecting track fragment relationships.
*   **video_splicing.py**: Generates "always-centered" highlight clips for individual tracks.
    *   Normalizes subject size based on a target bounding box height ratio.
    *   Applies Exponential Moving Average (EMA) smoothing to zoom/scale factors.
    *   Exports fixed-resolution MP4s for specific track IDs.

### Subfolders

*   **debug/**: Interactive tools for inspecting Kalman filter states, cost matrices, and track fragment linking.

### TODO

*   Implement current frame transform fetching in `CameraMotionTrailOverlay.apply`.
*   Refine 'DENSE' mode placeholder in `compute_stabilization_transforms`.
*   Standardize frame index tracking within `Overlay` objects.
