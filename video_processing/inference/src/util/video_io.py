from __future__ import annotations

import os
import subprocess
import cv2
import time
from typing import Generator
from dataclasses import dataclass


@dataclass
class VideoInfo:
    fps: int
    width: int
    height: int
    approximate_total_frames: int


@dataclass
class AccurateVideoInfo(VideoInfo):
    total_frames: int


class VideoReader:
    def __init__(self, video_path: os.PathLike | str, drop_every_nth: int = 1):
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
            approximate_total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
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


def get_video_properties(video_path: os.PathLike | str) -> VideoInfo:
    with VideoReader(video_path) as reader:
        return reader.get_properties()


def get_video_properties_with_accurate_total_frame_count(video_path: os.PathLike | str) -> AccurateVideoInfo:
    """
    Return VideoInfo where `approximate_total_frames` is replaced with an ffprobe-derived frame count.

    We keep the field name `approximate_total_frames` to preserve existing call sites that expect a cheap
    `get_video_properties()` call. Use this helper only in places that must have a correct count (e.g.
    stabilization arrays, time_percent computations, per-frame loops).
    """
    props = get_video_properties(video_path)
    accurate_total_frames = get_video_total_frame_count(video_path)
    return AccurateVideoInfo(
        fps=int(props.fps),
        width=int(props.width),
        height=int(props.height),
        approximate_total_frames=int(accurate_total_frames),
        total_frames=int(accurate_total_frames),
    )


def get_video_total_frame_count(video_path: os.PathLike | str) -> int:
    """
    Return the best-available total frame count for `video_path` without decoding frames.

    OpenCV's `CAP_PROP_FRAME_COUNT` is often wrong for VFR MP4s and some codecs/containers.
    We prefer `ffprobe` packet inspection to match how the web player derives its frame count:
    mediabunny enumerates encoded packets and ignores packets with negative timestamps.
    """
    # Demux-only (no decode). Count only packets with non-negative PTS (presentable packets),
    # matching `getSortedVideoPacketMeta` which filters `pkt.timestamp < 0`.
    cmd_packets_pts = [
        'ffprobe',
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_packets',
        '-show_entries',
        'packet=pts_time',
        '-of',
        'csv=p=0',
        str(video_path),
    ]
    try:
        res = subprocess.run(cmd_packets_pts, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise RuntimeError('ffprobe not found on PATH; required for accurate frame counts') from e

    if res.returncode != 0:
        msg = (res.stderr or res.stdout or '').strip()
        raise RuntimeError(f'ffprobe failed (exit {res.returncode}) for {video_path}: {msg}')

    raw = (res.stdout or '').strip()
    count = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line or line == 'N/A':
            continue
        try:
            pts = float(line)
        except ValueError:
            continue
        if pts >= 0:
            count += 1
    if count > 0:
        return count

    # Fallback: stream-level packet count (still demux-only), but does not exclude negative PTS packets.
    cmd_count_packets = [
        'ffprobe',
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-count_packets',
        '-show_entries',
        'stream=nb_read_packets',
        '-of',
        'csv=p=0',
        str(video_path),
    ]
    res2 = subprocess.run(cmd_count_packets, capture_output=True, text=True, check=False)
    if res2.returncode != 0:
        msg = (res2.stderr or res2.stdout or '').strip()
        raise RuntimeError(f'ffprobe failed (exit {res2.returncode}) for {video_path}: {msg}')
    for line in (res2.stdout or '').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            n = int(line)
        except ValueError:
            continue
        if n > 0:
            return n

    raise RuntimeError(f'ffprobe returned no usable packet count for {video_path}')


class VideoWriter:
    def __init__(self, output_path: os.PathLike | str, width: int, height: int, fps: int, fourcc: str = 'mp4v'):
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


class LiveVideoStreamer:
    """
    Context-manager that streams video frames to an OpenCV window at (≈) the
    requested FPS.  Use exactly as you would a file/video writer:

        with LiveVideoStreamer(w, h, fps) as streamer:
            streamer.write_frame(frame)

    -- Parameters --
    width, height : int
        Expected frame size (frames are auto-resized if they don’t match).
    fps : float | int
        Target playback speed.  A value ≤ 0 disables pacing (fast as possible).

    -- Methods --
    write_frame(frame) -> bool
        Displays *frame* and returns *True* unless the user pressed “q”.
        Check the return value if you want to support early quit.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        window_name: str = 'Live Stream',
        flags: int = cv2.WINDOW_NORMAL,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.window_name = window_name
        self.flags = flags

        self._delay_ms: int = 1 if self.fps <= 0 else max(1, int(1000 / self.fps))
        self._next_frame_time: float = time.perf_counter()

    # ───────────────────────── context-manager hooks ────────────────────────── #

    def __enter__(self) -> LiveVideoStreamer:
        cv2.namedWindow(self.window_name, self.flags)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv2.destroyWindow(self.window_name)

    # ───────────────────────────── public API ───────────────────────────────── #

    def write_frame(self, frame) -> bool:
        """
        Show *frame* and pace playback.  Returns False iff the user pressed “q”.
        """
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        cv2.imshow(self.window_name, frame)

        # Wait the minimal time so the window can refresh & catch key events
        key = cv2.waitKey(self._delay_ms) & 0xFF
        if key == ord('q'):
            return False

        # If FPS pacing is enabled, sleep until the next tick
        if self.fps > 0:
            self._next_frame_time += 1.0 / self.fps
            now = time.perf_counter()
            sleep_s = self._next_frame_time - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:  # We fell behind; reset reference
                self._next_frame_time = now

        return True
