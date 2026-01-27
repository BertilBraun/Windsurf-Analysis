# Windsurfing Video Player Interface - Requirements Document

Status: **partially implemented / partially historical**.

This doc was written as a forward-looking spec for the interactive player. The current web player exists in
`frontend/src/ui/player/…`, but some details below (file formats, exact output structure, implementation language)
may not match what’s shipped today.

Notable differences vs the current system:

- The production pipeline uses a **YOLO pose** model (bbox + 2 keypoints) and computes per-frame `anchor` + `scale` for focused view stability.
- ReID/appearance is configurable; the default in production is not necessarily OSNet.
- The web app stores job results as JSON in Firebase Storage/GCS (rather than generating per-track video files by default).

For the up-to-date implementation overview, start with:

- `frontend/public/TECHNICAL.md`
- `documentation/README.md`

## Overview

This document outlines the requirements for a new interactive video player interface for the Windsurfing Video Analysis system. The player will allow users to efficiently review and analyze processed windsurfing videos by seamlessly switching between overview and detailed views of individual surfers.

## Current System Architecture

### Processing Pipeline

The current system processes videos through the following stages:

1. **Video Input**: Raw windsurfing videos (`.mp4` files)
2. **Object Detection**: Uses YOLO model to detect windsurfers in each frame
3. **Re-identification**: Uses OSNet model to generate embeddings for detected surfers
4. **Tracking**: Multiple tracking algorithms to create consistent tracks:
   - GreedyPreprocessor: Initial track preprocessing
   - DiscreteILPTracker: Optimization-based track linking
   - TrackFilteringSmoothingRelabeling: Final track refinement
5. **Video Generation**: Creates two types of output videos:
   - **Annotated videos**: Original video with bounding boxes and track IDs overlaid
   - **Individual videos**: Cropped and centered videos focused on each surfer

### Current Output Structure

For each processed video `{input_name}`, the system generates:

- `{input_name}+00_annotated.mp4`: Overview video with all tracks annotated
- `{input_name}+{track_id:02d}.mp4`: Individual surfer videos (one per track)
- `{input_name}+{track_id:02d}.start_time.json`: Metadata with track start time

### Data Structures

- **Track**: Contains track_id and sorted_detections list
- **Detection**: Contains bbox, embedding, confidence, frame_idx, color_histogram
- **BoundingBox**: Contains x1, y1, x2, y2 coordinates with utility methods
- **VideoInfo**: Contains fps, width, height, total_frames

## New Player Interface Requirements

### 1. Architecture Overview

The new player interface should be designed as a separate application that can load and display processed video data without re-running the expensive detection and tracking pipeline.

#### 1.1 Separation of Concerns

- **Processing Pipeline**: Existing system focuses solely on detection, tracking, and video generation
- **Player Interface**: New system focuses on interactive playback and analysis
- **Data Exchange**: Serialized track data and metadata files bridge the two systems

#### 1.2 Performance Optimization

- The player must be responsive and efficient since the goal is to optimize the analysis workflow
- Video loading and switching should be near-instantaneous
- The interface should support smooth playback at various speeds

### 2. Data Management

#### 2.1 Metadata Serialization

The processing pipeline must be modified to save track data for later loading:

**Track Metadata File** (`{input_name}.tracks.json`):

```json
{
  "input_video_path": "path/to/original/video.mp4",
  "video_properties": {
    "fps": 30,
    "width": 1920,
    "height": 1080,
    "total_frames": 9000
  },
  "tracks": [
    {
      "track_id": 0,
      "start_frame": 150,
      "end_frame": 8950,
      "start_time": 5.0,
      "duration": 293.33,
      "detection_count": 1250,
      "detections": [
        {
          "frame_idx": 150,
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.95,
          "embedding": [array of floats],
          "color_histogram": [array of floats]
        }
        // ... more detections
      ]
    }
    // ... more tracks
  ]
}
```

Remove all other files which were written by the processing pipeline like the individual videos and the annotated video as well as the start_time.json files.

### 3. User Interface Requirements

#### 3.1 Core Interaction Modes

**Overview Mode** (Default):

- Displays the annotated video. I.e. the original video with all bounding boxes and track IDs overlaid (see @visualization/annotation_drawer.py)
- Shows all tracks with colored bounding boxes and track IDs
- Allows clicking on any surfer to switch to detailed view
- Track list shows all available tracks with checkboxes for visibility

**Detailed Mode**:

- Displays individual surfer video I.e. the original video zoomed and cropped to the bounding box of the selected surfer (see @visualization/video_splicing.py)
- Shows cropped, centered, and scaled view of the selected surfer
- Maintains temporal synchronization with the overview video
- Press Escape to return to overview mode

#### 3.2 Video Playback Controls

**Basic Controls**:

- Play/Pause (Spacebar)
- Step forward/backward by frame (Arrow keys)
- Seek backward/forward by 5 seconds (Left/Right + Shift)
- Seek backward/forward by 30 seconds (Left/Right + Ctrl)

**Speed Controls**:

- Playback speed options: 0.25x, 0.5x, 1x, 2x, 4x, 8x with hotkeys + and -
- Smooth speed transitions
- Speed indicator in UI as a short text popup once changed for 1 second

**Zoom Controls**:

- Use the mouse position to zoom in and out on that position
- Use the mouse wheel to zoom in and out
- Zoom indicator in UI as a short text popup once changed for 1 second

**Timeline Interaction**:

- Clickable timeline bar for direct seeking
- Hover preview (optional enhancement)
- Track duration indicators on timeline
- Current position indicator (time in seconds)

#### 3.3 Track Management

**Track Selection**:

- Click on surfer in overview mode to enter detailed mode
- Color-coded track identifiers consistent with annotations

#### 3.4 Video Loading

- Allow the user to go through all videos in the output directory
- Go to the next video in the directory by pressing 'n'
- Go to the previous video in the directory by pressing 'p'
- Do not automatically load the next video when the current one ends
- Display the current video name in the UI (title bar)

### 4. Technical Implementation Requirements

#### 4.1 Framework Selection

**Recommended**: Python with one of the following GUI frameworks:

- **PyQt6/PySide6**: Professional, mature, excellent video support
- **Tkinter + opencv-python**: Lightweight, already in dependencies
- **Kivy**: Modern, touch-friendly, good performance
- **Web-based (FastAPI + HTML5)**: Browser-based, platform independent

#### 4.2 Video Playback Engine

- Use OpenCV (`cv2.VideoCapture`) for video decoding (already in dependencies)
- Implement frame caching for smooth playback
- Support variable playback speeds
- Handle video format compatibility (MP4/H.264)

#### 4.3 Data Loading

**Video Loading**:

- Pre-load video properties (fps, frame count, dimensions)
- Efficient frame seeking and buffering
- Memory management for multiple video files

#### 4.4 State Management

```python
class PlayerState:
    current_mode: Literal["overview", "detailed"]
    current_track_id: Optional[int]
    current_frame: int
    playback_speed: float
    is_playing: bool
    loaded_tracks: List[Track]
    visible_tracks: Set[int]
    video_properties: VideoInfo
```

### 5. File Structure for New Player

```
src/
├── player/
│   ├── __init__.py
│   ├── main.py                    # Entry point for player application
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── main_window.py         # Main application window
│   │   ├── video_widget.py        # Video display component
│   │   ├── controls_widget.py     # Playback controls
│   │   ├── timeline_widget.py     # Timeline scrubber
│   │   └── track_list_widget.py   # Track management panel
│   ├── core/
│   │   ├── __init__.py
│   │   ├── player_state.py        # Application state management
│   │   ├── video_manager.py       # Video loading and playback
│   │   ├── metadata_loader.py     # Track data loading
│   │   └── frame_cache.py         # Frame caching for performance
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py         # Video-related utilities
│       └── ui_utils.py            # UI helper functions
```

### 6. Integration with Existing System

#### 6.1 Processing Pipeline Modifications

**Add to `windsurf_video_processor.py`**:

```python
def _save_tracks_metadata(tracks: List[Track], input_path: Path, output_dir: Path, video_props: VideoInfo):
    """Save track metadata for later loading by the player interface"""
    metadata = {
        "input_video_path": str(input_path),
        "video_properties": {
            "fps": video_props.fps,
            "width": video_props.width, 
            "height": video_props.height,
            "total_frames": video_props.total_frames
        },
        "tracks": [
            {
                "track_id": track.track_id,
                "start_frame": track.start_frame(),
                "end_frame": track.end_frame(),
                "start_time": track.start_frame() / video_props.fps,
                "duration": (track.end_frame() - track.start_frame()) / video_props.fps,
                "detection_count": len(track.sorted_detections),
                "detections": [
                    {
                        "frame_idx": det.frame_idx,
                        "bbox": [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                        "confidence": float(det.confidence),
                        "embedding": det.embedding.tolist(),
                        "color_histogram": det.color_histogram.tolist()
                    }
                    for det in track.sorted_detections
                ]
            }
            for track in tracks
        ]
    }
    
    metadata_path = output_dir / f"{input_path.stem}.tracks.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
```


### 7. User Experience Flow

#### 7.1 Typical Usage Scenario

1. **Launch Player**: User runs player and selects output directory
2. **Load Session**: Player scans directory for `.tracks.json` files
3. **The player automatically loads the first video in the directory**
4. **Overview Analysis**: User plays annotated video to see all surfers
5. **Track Selection**: User clicks on interesting surfer to switch to detailed view
6. **Detailed Analysis**: User analyzes individual surfer performance
7. **Quick Switching**: User presses Escape to return to overview, clicks different surfer
8. **Timeline Navigation**: User uses timeline, keyboard shortcuts and controls for efficient navigation
9. **The user can go to the next video in the directory by pressing 'n'**

#### 7.2 Keyboard Shortcuts

```
Spacebar    - Play/Pause
Left/Right  - Step backward/forward by frame
Shift+Left/Right - Seek backward/forward by 5 seconds  
Ctrl+Left/Right  - Seek backward/forward by 30 seconds
Escape      - Return to overview mode (from detailed mode)
+           - Increase playback speed
-           - Decrease playback speed
Q           - Quit application
n           - Go to the next video in the directory
p           - Go to the previous video in the directory
mouse_wheel - Zoom in and out on the mouse position
mouse_click - Go to the detailed view of the surfer under the mouse cursor
```
