from typing import Generator
import cv2
import os
from dataclasses import dataclass


@dataclass
class VideoInfo:
    fps: int
    width: int
    height: int
    total_frames: int


class VideoReader:
    def __init__(self, video_path: os.PathLike, drop_every_nth: int = 1):
        self.video_path = video_path
        self.drop_every_nth = drop_every_nth
        self.cap = None
        self.frame_index = 0

    def __enter__(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f'Cannot open video file: {self.video_path}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def get_properties(self) -> VideoInfo:
        """Get video properties"""
        assert self.cap is not None, 'VideoCapture not initialized'
        return VideoInfo(
            fps=int(self.cap.get(cv2.CAP_PROP_FPS)),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def read_frames(self) -> Generator[tuple[int, cv2.typing.MatLike], None, None]:
        """Generator that yields frames with optional frame dropping"""
        assert self.cap is not None, 'VideoCapture not initialized'
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Only yield frame if we're not dropping it
            if self.frame_index % self.drop_every_nth == 0:
                yield self.frame_index, frame

            self.frame_index += 1


def get_video_properties(video_path: os.PathLike):
    with VideoReader(video_path) as reader:
        return reader.get_properties()


class VideoWriter:
    def __init__(self, output_path: os.PathLike, width: int, height: int, fps: int, fourcc: str = 'mp4v'):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = cv2.VideoWriter.fourcc(*fourcc)
        self.writer = None

    def start_writing(self):
        # Create output directory if needed
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.writer = cv2.VideoWriter(str(self.output_path), self.fourcc, self.fps, (self.width, self.height))

        if not self.writer.isOpened():
            raise ValueError(f'Cannot create output video file: {self.output_path}')

    def finish_writing(self):
        if self.writer:
            self.writer.release()

    def __enter__(self):
        self.start_writing()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish_writing()

    def write_frame(self, frame: cv2.typing.MatLike) -> None:
        """Write a single frame to the video"""
        assert self.writer is not None, 'VideoWriter not initialized'
        self.writer.write(frame)
