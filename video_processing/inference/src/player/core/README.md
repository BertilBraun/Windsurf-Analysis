# Player Core

Core logic and state management for video playback and inference visualization.

### Components

*   **PlayerState**: Central manager for playback and detection state.
    *   Tracks playback status: current frame, speed, and play/pause state.
    *   Manages playback modes: overview and detailed.
    *   Maintains a fast lookup index mapping frame indices to detections.
    *   Stores and retrieves frame-by-frame stabilization data (translation and rotation).
    *   Handles track visibility and metadata.
*   **VideoManager**: OpenCV wrapper for frame-accurate video access.
    *   Supports seeking to specific frame indices.
    *   Provides efficient frame skipping using grab() to minimize decoding overhead.
    *   Extracts video properties: FPS, dimensions, and total frame count.

### Data Models

*   **DetectionLite**: Lightweight container for detection data.
    *   Includes bounding boxes, confidence, and interpolation flags.
    *   Stores pose keypoints: boom and mast_tip (coordinates + confidence).
    *   Contains precomputed rendering fields: anchor (x, y) and scale.
*   **TrackLite**: Groups detections by track ID with temporal metadata (start/end frames and duration).
*   **VideoProperties**: Metadata container for video dimensions and timing.
*   **Metadata**: Top-level container linking a video path to its properties and tracks.

### TODO

*   Implement logic for switching between overview and detailed modes within PlayerState.
*   Integrate real-time stabilization data updates if provided by external pipelines.
