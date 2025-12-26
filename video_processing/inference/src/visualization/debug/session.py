from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np

from ...visualization.debug.draw import compose_side_by_side
from ...visualization.debug.viewer import ViewerCV2, ViewerInterface


class Overlay(Protocol):
    def apply(self, image: np.ndarray) -> None: ...


class DebugSession(Protocol):
    def close(self) -> None: ...
    def show_frame(
        self, frame_index: int, overlays: Optional[List[Overlay]] = None, hud_text: Optional[str] = None
    ) -> None: ...
    def show(self, *images, window_name: str = '') -> None: ...
    def get_frame(self, frame_index: int) -> Optional[np.ndarray]: ...
    def set_mouse_callback(
        self, window_name: str, callback: Callable[[int, int, int, int, object], None] | None
    ) -> None: ...
    def scroll(self, start_frame_index: int = 0) -> None: ...
    def hud(self, message: str) -> None: ...
    def wait_step(self) -> int: ...


class NullDebugSession(DebugSession):
    """No-op debug session used when debug is disabled.

    All public methods are present but perform no actions.
    """

    def close(self) -> None:
        return

    # Factory compatibility
    @staticmethod
    def from_video(video_path: Path) -> NullDebugSession:
        return NullDebugSession()

    # API stubs
    def show_frame(
        self, frame_index: int, overlays: Optional[List[Overlay]] = None, hud_text: Optional[str] = None
    ) -> None:
        return

    def show(self, *images, window_name: str = '') -> None:
        return

    def set_mouse_callback(
        self, window_name: str, callback: Callable[[int, int, int, int, object], None] | None
    ) -> None:
        return

    def scroll(self, start_frame_index: int = 0) -> None:
        return

    def hud(self, message: str) -> None:
        return

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        return None

    def wait_step(self) -> int:
        return 0


class Cv2DebugSession(DebugSession):
    """Active debug session that owns frames, viewers, and configuration."""

    def __init__(self, frames: Dict[int, np.ndarray]) -> None:
        self._frames: Dict[int, np.ndarray] = frames
        self._closed: bool = False

        # Lazy attributes for heavy backends; instantiated on first use
        self._viewer: Optional[ViewerInterface] = None

    @staticmethod
    def from_video(video_path: Path) -> Cv2DebugSession:
        # Lazy import to avoid cv2 in hot path when disabled elsewhere
        from ...util.video_io import VideoReader  # type: ignore

        frames: Dict[int, np.ndarray] = {}
        with VideoReader(video_path.as_posix()) as reader:
            for frame_index, frame in reader.read_frames():
                frames[int(frame_index)] = frame
        return Cv2DebugSession(frames=frames)

    # --------------- Public API --------------- #
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._viewer is not None:
                self._viewer.destroy_all()
        finally:
            self._frames.clear()

    def show_frame(
        self, frame_index: int, overlays: Optional[List[Overlay]] = None, hud_text: Optional[str] = None
    ) -> None:
        if self._closed:
            return

        frame = self.get_frame(frame_index)
        if frame is None:
            return
        image = frame.copy()
        if overlays:
            for overlay in overlays:
                overlay.apply(image)

        viewer = self._get_viewer()
        viewer.show('frame', image, hud_text=hud_text)

    def show(self, *images, window_name: str = '') -> None:
        if self._closed:
            return
        viewer = self._get_viewer()
        viewer.show(window_name, compose_side_by_side(*images))

    def set_mouse_callback(
        self, window_name: str, callback: Callable[[int, int, int, int, object], None] | None
    ) -> None:
        if self._closed:
            return
        viewer = self._get_viewer()
        # ViewerCV2 supports this; keep optional to avoid tightening interface too much.
        if hasattr(viewer, 'set_mouse_callback'):
            viewer.set_mouse_callback(window_name, callback)  # type: ignore[attr-defined]

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        return self._frames.get(int(frame_index))

    def scroll(self, start_frame_index: int = 0) -> None:
        if self._closed:
            return
        viewer = self._get_viewer()
        viewer.scroll(self._frames, start_frame_index)

    def hud(self, message: str) -> None:
        if self._closed:
            return
        viewer = self._get_viewer()
        viewer.set_hud(message)

    def wait_step(self) -> int:
        """Block until key press and return navigation step (-1 left, +1 right/any, 0 quit)."""
        if self._closed:
            return 0
        viewer = self._get_viewer()
        return int(viewer.wait_step())

    # --------------- internals --------------- #

    def _get_viewer(self) -> ViewerInterface:
        if self._viewer is None:
            self._viewer = ViewerCV2()

        return self._viewer
