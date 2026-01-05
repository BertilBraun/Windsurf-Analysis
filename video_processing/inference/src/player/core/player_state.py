from __future__ import annotations

from dataclasses import dataclass
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
    interpolated: bool


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


class PlayerState:
    def reset(self, input_video_path: str, video_properties: VideoProperties, loaded_tracks: List[TrackLite]) -> None:
        self.input_video_path = input_video_path
        self.video_properties = video_properties
        self.loaded_tracks = loaded_tracks

        self.current_mode: Literal['overview', 'detailed'] = 'overview'
        self.current_track_id: Optional[int] = None
        self.current_frame: int = 0
        self.playback_speed: float = 1.0
        self.is_playing: bool = False

        # Fast lookup of visible track ids
        self.visible_tracks = self._extract_visible_tracks()
        # Fast lookup of detections that occur at a given frame index
        self.detections_by_frame = self._rebuild_detection_index()

    def _extract_visible_tracks(self) -> Set[int]:
        return {t.track_id for t in self.loaded_tracks}

    def _rebuild_detection_index(self) -> Dict[int, List[Tuple[int, DetectionLite]]]:
        """Builds a dictionary mapping frame index to detections present at that frame.

        The value list contains tuples of (track_id, DetectionLite).
        """
        index: Dict[int, List[Tuple[int, DetectionLite]]] = {}
        for track in self.loaded_tracks:
            for det in track.detections:
                index.setdefault(det.frame_idx, []).append((track.track_id, det))
        return index
