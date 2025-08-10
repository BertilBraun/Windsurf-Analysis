from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Set, List, Dict, Tuple


@dataclass
class VideoProperties:
    fps: float
    width: int
    height: int
    total_frames: int


@dataclass
class DetectionLite:
    frame_idx: int
    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float


@dataclass
class TrackLite:
    track_id: int
    start_frame: int
    end_frame: int
    start_time: float
    duration: float
    detection_count: int
    detections: List[DetectionLite]


@dataclass
class Metadata:
    input_video_path: str
    video_properties: VideoProperties
    tracks: List[TrackLite]


@dataclass
class PlayerState:
    current_mode: Literal['overview', 'detailed'] = 'overview'
    current_track_id: Optional[int] = None
    current_frame: int = 0
    playback_speed: float = 1.0
    is_playing: bool = False
    loaded_tracks: List[TrackLite] = field(default_factory=list)
    visible_tracks: Set[int] = field(default_factory=set)
    video_properties: Optional[VideoProperties] = None
    input_video_path: Optional[str] = None
    # Fast lookup of detections that occur at a given frame index
    detections_by_frame: Dict[int, List[Tuple[int, DetectionLite]]] = field(default_factory=dict)

    def reset_for_new_video(self) -> None:
        self.current_mode = 'overview'
        self.current_track_id = None
        self.current_frame = 0
        self.playback_speed = 1.0
        self.is_playing = False
        self.loaded_tracks = []
        self.visible_tracks = set()
        self.video_properties = None
        self.input_video_path = None
        self.detections_by_frame = {}

    def rebuild_detection_index(self) -> None:
        """Builds a dictionary mapping frame index to detections present at that frame.

        The value list contains tuples of (track_id, DetectionLite).
        """
        index: Dict[int, List[Tuple[int, DetectionLite]]] = {}
        for track in self.loaded_tracks:
            for det in track.detections:
                index.setdefault(det.frame_idx, []).append((track.track_id, det))
        self.detections_by_frame = index
