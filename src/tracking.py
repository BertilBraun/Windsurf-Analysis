from video_io import VideoInfo
from common_types import Detection, Track


from typing import Protocol


class Tracker(Protocol):
    def track_detections(self, detections: list[Detection], video_properties: VideoInfo) -> list[Track]: ...


class DummyTracker:
    def track_detections(self, detections: list[Detection], video_properties: VideoInfo) -> list[Track]:
        return [
            Track(
                track_id=None,
                sorted_detections=[detection],
            )
            for detection in detections
        ]
