from vidstab import VidStab

from video_io import VideoInfo
from common_types import Detection, Track


def process_detections(detections: list[Detection], video_properties: VideoInfo, stabilizer: VidStab) -> list[Track]:
    return [
        Track(
            track_id=None,
            sorted_detections=[detection],
        )
        for detection in detections
    ]
