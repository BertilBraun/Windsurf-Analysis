from common_types import Detection, FrameIndex, Track
from video_io import VideoInfo
from track_processing import TrackerInput


def process_detections(detections: dict[FrameIndex, list[Detection]], video_properties: VideoInfo) -> TrackerInput:
    return {
        1: [
            Track(
                bbox=detection.bbox,
                confidence=detection.confidence,
                track_id=None,
                feat=detection.feat,
                frame_idx=frame_idx,
            )
            for frame_idx, detections in detections.items()
            for detection in detections
        ]
    }
