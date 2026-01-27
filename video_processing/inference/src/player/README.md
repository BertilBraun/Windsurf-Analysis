# Player

PySide6-based application for visualizing video inference results, object tracks, and stabilization data.

### Subfolders

*   **[core](./core)**: Logic and state management. Handles video decoding via OpenCV, playback state, and data models for detections and tracks.
*   **[ui](./ui)**: User interface components. Includes the main window, video rendering surface (with stabilization support), timeline, and playback controls.

### Key Features

*   **Dual Viewing Modes**:
    *   **Overview**: Full-frame view with optional motion compensation (rotation/translation).
    *   **Detailed**: Automatically crops and centers on a specific track anchor to eliminate jitter.
*   **Inference Visualization**: Renders bounding boxes (color-coded for interpolation), keypoints (boom, mast tip), and track anchors.
*   **Frame-Accurate Navigation**: Supports single-frame stepping, time-based jumps, and variable playback speeds (0.25x to 8x).
*   **Stabilization Integration**: Applies per-frame transforms from JSON metadata to counter camera shake in real-time.
*   **Interactive Controls**: Mouse-based zooming/panning and a comprehensive set of keyboard shortcuts for playback control.

### Data Requirements

*   **Video**: Standard formats supported by OpenCV.
*   **Tracks**: `.tracks.pkl` files containing `Metadata` objects (tracks, video properties).
*   **Stabilization**: `<filename>.stabilization_transforms.json` containing per-frame `dx`, `dy`, and `da` values.

### TODO

*   Implement logic for switching between overview and detailed modes within the central state manager.
*   Add support for exporting cropped clips directly from the interface.
*   Develop a track sidebar for advanced multi-track filtering.
*   Integrate real-time stabilization data updates from external pipelines.
