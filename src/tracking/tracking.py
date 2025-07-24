from video_io import VideoInfo
from common_types import Track
from typing import Protocol


class Tracker(Protocol):
    def track(self, tracks: list[Track], video_properties: VideoInfo) -> list[Track]: ...
