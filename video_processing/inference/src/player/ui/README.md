PySide6-based user interface components for the Windsurf Player, designed to visualize video inference results, object tracks, and stabilization data.

### Core Components

*   **main_window.py**: The primary application window. Orchestrates `PlayerState`, `VideoManager`, and sub-widgets. Handles keyboard shortcuts, file I/O for metadata, and the main playback timer.
*   **video_widget.py**: The rendering surface for video frames and overlays.
    *   **Overview Mode**: Full-frame view with optional stabilization (rotation/translation), zooming, and panning.
    *   **Detailed Mode**: Automatically crops and centers the view on a specific track anchor, ignoring stabilization to reduce high-zoom jitter.
    *   **Overlays**: Renders bounding boxes (color-coded for interpolation), keypoints (boom, mast tip), and anchor indicators.
*   **timeline_widget.py**: A custom seek bar that visualizes the total video duration and highlights segments where object tracks are present.
*   **controls_widget.py**: UI buttons for play/pause, speed adjustment, and navigating between video files in a directory.

### Key Features

*   **Frame-Locked Playback**: Ensures precise frame stepping and synchronized metadata display.
*   **Stabilization Support**: Reads per-frame transforms (`dx`, `dy`, `da`) from JSON to counter camera shake in Overview mode.
*   **Interactive Tracking**: Click objects in overview mode to enter "Detailed" mode for that track; right-click to pan and scroll to zoom.
*   **Keyboard Shortcuts**:
    *   `Space`: Play/Pause.
    *   `Left/Right`: Single frame step.
    *   `Shift + Left/Right`: 5-second jump.
    *   `Ctrl + Left/Right`: 30-second jump.
    *   `+/-` or `=`: Adjust playback speed (0.25x to 8x).
    *   `Esc`: Return to overview mode.
    *   `N/P`: Load next or previous video in the directory.
    *   `Q`: Close application.

### Data Requirements

*   **Metadata**: Expects `.tracks.pkl` files containing `Metadata` objects (tracks, video properties).
*   **Stabilization**: Looks for `<filename>.stabilization_transforms.json` in the same directory for per-frame motion compensation.

### TODO

*   Implement a more robust track sidebar if complex multi-track filtering is required.
*   Add support for exporting cropped clips directly from the UI.
