This folder contains Python scripts for video processing, human pose estimation, and interactive annotation, primarily focused on windsurfing footage.

### Core Scripts

*   **pose_video.py**: Human pose detection using YOLOv8-pose.
    *   Supports full-frame inference, detector-based cropping for distant subjects, or fixed center-cropping.
    *   Filters results by keypoint confidence, torso visibility, and temporal persistence across frames.
    *   Outputs annotated videos and/or JSON files containing keypoint coordinates.
*   **ws_ingress.py**: Interactive workflow for reviewing and cutting raw footage.
    *   Uses `mpv` to preview videos and mark start/end points for clips.
    *   Prompts for tags via Zenity (GUI) or CLI to generate descriptive filenames.
    *   Handles cutting, rotation, and optional stabilization using parallel background workers.
*   **ws_reduce_size.py**: Batch video compression with two profiles:
    *   **send**: Targets a specific file size (e.g., 64MB) for messaging using 2-pass encoding.
    *   **keep**: High-quality archival encoding using CRF (Constant Rate Factor).
    *   Automatically selects between HEVC (x265) and H.264 based on system support.
*   **ws_stabilize.py**: Standalone utility for parallel video stabilization.
    *   Wraps FFmpeg's `vidstabdetect` and `vidstabtransform` filters.
    *   Supports batch processing with configurable parallel jobs and progress bars.

### Shared Utilities

*   **util.py**: Common helper functions used across the toolset.
    *   `MpvIPCTask`: Context manager to control `mpv` via JSON IPC.
    *   `stabilize_ffmpeg`: Standardized FFmpeg command for video smoothing.
    *   `setup_logging`: Configures logging with optional desktop notifications via `notify-send`.

### Requirements & Dependencies

*   **FFmpeg**: Required for all video manipulation (cutting, stabilizing, encoding).
*   **mpv**: Required for interactive annotation in `ws_ingress.py`.
*   **Ultralytics (YOLO)**: Required for `pose_video.py`.
*   **Zenity**: Optional; used for popup tag entry in `ws_ingress.py`.
*   **Python Libraries**: `opencv-python`, `torch`, `numpy`, `tqdm`.

### TODO

*   Add a `requirements.txt` or environment specification for Python dependencies.
*   Include the `mpv_script_cutting.lua` referenced by `ws_ingress.py`.
