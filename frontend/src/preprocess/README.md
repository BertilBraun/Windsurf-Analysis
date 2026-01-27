Utilities for client-side video transcoding and frame-by-frame manipulation using the Mediabunny library.

### Key Features

*   **Quality Presets**: Predefined targets for video re-encoding:
    *   `original`: Maintains source dimensions and frame rate.
    *   `high`: Targets up to 1080p at 30 FPS.
    *   `medium`: Targets up to 720p at 25 FPS.
    *   `minimum`: Targets 640px (long side) at 15 FPS.
*   **Frame-by-Frame Processing**: Custom canvas-based manipulation via `onFrame` callback using `OffscreenCanvas` or DOM canvas.
*   **Transcoding & Resizing**:
    *   Automatically fits video to target dimensions while maintaining aspect ratio.
    *   Ensures even dimensions (width/height) for codec compatibility.
    *   Outputs processed video as MP4 `ArrayBuffer`.
*   **Constant Frame Rate (CFR)**: Ensures stable output frame rates even if the source is variable or frames are dropped.
*   **Progress Tracking**: Supports callbacks to monitor processing progress from 0.0 to 1.0.

### Main Functions

*   **`preprocessVideo`**: High-level utility to resize and re-encode a video file based on a quality preset.
*   **`processVideo`**: Low-level utility for advanced workflows requiring custom drawing, filtering, or frame skipping on a per-frame basis.

### TODO

*   Consider using Mediabunny's built-in file conversion guides for simpler transcoding tasks that do not require canvas manipulation.
