from video_io import VideoInfo
from common_types import Detection, Track


class Tracker:
    def track_detections(self, detections: list[Detection], video_properties: VideoInfo) -> list[Track]:
        raise NotImplementedError("Subclasses should implement this method.")


class DummyTracker(Tracker):
    def track_detections(self, detections: list[Detection], video_properties: VideoInfo) -> list[Track]:
        return [
            Track(
                track_id=None,
                sorted_detections=[detection],
            )
            for detection in detections
        ]
