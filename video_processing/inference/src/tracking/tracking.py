from typing import Protocol

from ..util.video_io import VideoInfo
from ..common_types import Track
from ..visualization.stabilize import Transform


class Tracker(Protocol):
    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]: ...
