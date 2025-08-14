from typing import Protocol

from ..util.video_io import VideoInfo
from ..common_types import Track


class Tracker(Protocol):
    def track(self, tracks: list[Track], video_properties: VideoInfo) -> list[Track]: ...
