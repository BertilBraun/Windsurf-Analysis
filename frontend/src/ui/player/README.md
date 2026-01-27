This folder contains the video player implementation, responsible for high-performance frame rendering, track visualization, and video export.

### Core Components
*   **Player**: The main orchestrator managing playback state, keyboard shortcuts, and specialized viewing modes (Overview vs. Detailed/Stabilized).
*   **ControlsBar**: UI for play/pause, playback speed, zoom levels, and export triggers.
*   **Timeline**: Interactive seek bar for frame-accurate navigation.
*   **DrawOverlay**: UI controls for the annotation system, including tool selection (freehand/line), stroke width, and color.

### Hooks and Logic
*   **useWebCodexPlayer**: Low-level video playback engine using WebCodecs. Handles frame-accurate seeking, decode caching, and playback loops.
*   **useAnnotations**: Manages frame-specific drawing state, supporting persistent strokes and undo functionality.
*   **useOverviewPan**: Implements panning and boundary logic for zoomed-in overview views.
*   **useJobVideoSource**: Resolves video files from local directory handles or direct file objects based on job metadata.
*   **PlayerState**: Manages the logical state of the player, including track detections, stabilization transforms, and frame-to-percent conversions.

### Rendering and Export
*   **rendering.ts**: Core canvas drawing logic. Handles stabilization, track bounding boxes, and coordinate transformations between screen and video space.
*   **rotation.ts**: Utilities for quantizing and applying video orientation in 90-degree increments.
*   **renderMath.ts**: Layout calculations for "contain" fit strategies and aspect ratio maintenance.
*   **export.ts**: Logic for generating MP4 exports of specific tracks. Applies stabilization, dynamic cropping, and watermarking during the process.
*   **watermark.ts**: Best-effort loading and rendering of brand logos onto exported videos.

### Constants
*   Defines crop limits (`MIN_CROP_NORM`, `MAX_CROP_NORM`) and default zoom baselines for detailed track views.

### TODO
*   Implement more robust error handling for WebCodecs initialization failures.
*   Optimize frame cache eviction strategy in `useWebCodexPlayer` for long videos.
