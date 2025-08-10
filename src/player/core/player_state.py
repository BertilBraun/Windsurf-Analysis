from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Set, List, Dict, Any


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
