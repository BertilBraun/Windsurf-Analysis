from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import cv2


class VideoManager:
    """Thin wrapper over OpenCV capture to support frame-accurate seek and read."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0.0
        self.total_frames: int = 0
        self.width: int = 0
        self.height: int = 0
        self.open()

    def open(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f'Failed to open video: {self.video_path}')
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def seek_frame(self, index: int) -> None:
        index = max(0, min(index, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    def read_frame(self) -> tuple[int, Optional[any]]:
        if self.cap is None:
            return -1, None
        ok, frame = self.cap.read()
        if not ok:
            return -1, None
        idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return idx, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
