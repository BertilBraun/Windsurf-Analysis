from __future__ import annotations

"""Reusable debug visualization package.

This package provides typed, reusable components to visualize frames, tracks,
detections, Kalman filters, camera motion trails, heatmaps, and interactive
graphs. It is designed to be imported without side effects and with lazy
imports for GUI/backends so production hot paths incur no overhead when debug
mode is disabled.
"""
from pathlib import Path
from typing import Optional

from .session import DebugSession, NullDebugSession, Cv2DebugSession


def get_debug_session(video_path: str | Path, enabled: Optional[bool] = None) -> DebugSession:
    if not enabled:
        return NullDebugSession()
    return Cv2DebugSession.from_video(video_path=Path(video_path))
