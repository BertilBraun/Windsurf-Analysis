from __future__ import annotations

import cv2

import numpy as np
from typing import Callable, Dict, Optional, Protocol


class ViewerInterface(Protocol):
    def destroy_all(self) -> None: ...
    def show(self, window_name: str, image: np.ndarray, hud_text: Optional[str] = None) -> None: ...
    def set_hud(self, message: str) -> None: ...
    def scroll(self, frames: Dict[int, np.ndarray], start_frame_index: int) -> None: ...
    def wait_step(self) -> int: ...


class ViewerCV2(ViewerInterface):
    """Thin wrapper around cv2 windows with basic HUD and scrolling support.

    All imports of cv2 are local to methods to avoid hard dependency in hot
    paths when debug is disabled.
    """

    def __init__(self) -> None:
        self._hud_text: str | None = None
        # Optional per-window mouse callbacks (enables interactive inspection in debug views)
        self._mouse_callbacks: Dict[str, Callable[[int, int, int, int, object], None]] = {}

    def destroy_all(self) -> None:
        cv2.destroyAllWindows()
        self._mouse_callbacks.clear()

    def show(self, window_name: str, image: np.ndarray, hud_text: str | None = None) -> None:
        to_show = image
        if hud_text:
            self._hud_text = hud_text
            cv2.putText(
                to_show,
                hud_text,
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, to_show)
        # If a mouse callback exists for this window, bind it (safe to call repeatedly)
        cb = self._mouse_callbacks.get(window_name)
        if cb is not None:
            cv2.setMouseCallback(window_name, cb)
        cv2.waitKey(1)

    def set_hud(self, message: str) -> None:
        self._hud_text = message

    def set_mouse_callback(self, window_name: str, callback: Callable[[int, int, int, int, object], None] | None) -> None:
        """
        Register an OpenCV mouse callback for a named window.
        Callback signature matches cv2.setMouseCallback: (event, x, y, flags, param).
        """
        if callback is None:
            self._mouse_callbacks.pop(window_name, None)
        else:
            self._mouse_callbacks[window_name] = callback
            # If the window already exists (e.g., callback registered right after first show),
            # bind immediately so clicks work without requiring another show() call.
            try:
                cv2.setMouseCallback(window_name, callback)
            except Exception:
                # Safe to ignore: window may not exist yet on some backends.
                pass

    def scroll(self, frames: Dict[int, np.ndarray], start_frame_index: int) -> None:
        if not frames:
            return

        frame_indices = sorted(frames.keys())
        if not frame_indices:
            return

        current = min(max(start_frame_index, frame_indices[0]), frame_indices[-1])

        def seek_to(new_index: int) -> None:
            nonlocal current
            current = min(max(new_index, frame_indices[0]), frame_indices[-1])
            img = frames[current]
            self.show('scroll', img, self._hud_text)

        seek_to(current)
        fps_guess = 30.0
        step_seconds = float(0.5)
        step_frames_seconds = int(round(fps_guess * step_seconds))

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q'), ord(' '), 13):
                break
            elif key == 81:  # Left
                seek_to(current - 1)
            elif key == 83:  # Right
                seek_to(current + 1)
            elif key == ord(','):
                seek_to(current - 30)
            elif key == ord('.'):
                seek_to(current + 30)
            elif key == ord('['):
                seek_to(current - step_frames_seconds)
            elif key == ord(']'):
                seek_to(current + step_frames_seconds)

    def wait_step(self) -> int:
        """Block until a key is pressed and return step delta: -1 for left, +1 otherwise, 0 on quit."""
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            return 0
        if key == 81 or key == ord(','):
            return -1
        return 1
