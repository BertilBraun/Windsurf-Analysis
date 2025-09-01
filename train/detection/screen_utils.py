#!/usr/bin/env python3
"""
screen_utils.py – shared utilities for screen size detection and overlays
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple


def get_screen_size() -> Optional[Tuple[int, int]]:
    """Return screen size as (width, height) in pixels, or None if unavailable."""
    # Prefer tkinter for cross-platform size; fallback to Windows API
    try:
        import tkinter as tk  # type: ignore

        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        return (w, h)
    except Exception:
        pass
    try:
        import ctypes  # type: ignore

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        w = int(user32.GetSystemMetrics(0))
        h = int(user32.GetSystemMetrics(1))
        return (w, h)
    except Exception:
        return None


def overlay_screen_warning(
    image: np.ndarray,
    screen_size: Optional[Tuple[int, int]],
    margins: Tuple[int, int] = (160, 220),
) -> np.ndarray:
    """Overlay a prominent red warning if image exceeds screen bounds.

    - image: BGR canvas (modified in place and also returned)
    - screen_size: (width, height) or None to skip
    - margins: tolerated margins (w_margin, h_margin)
    """
    if screen_size is None:
        return image
    H, W = image.shape[:2]
    screen_w, screen_h = screen_size
    margin_w, margin_h = margins
    if W <= max(1, screen_w - margin_w) and H <= max(1, screen_h - margin_h):
        return image

    warning = 'SCREEN TOO SMALL FOR IMAGE'
    subline = f'Image {W}x{H}, Screen {screen_w}x{screen_h}'

    # Dynamic font scaling
    base = max(1.0, min(W, H) / 600.0)
    font_scale = min(3.5, base * 2.0)
    thickness = max(2, int(font_scale * 3))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Main line centered
    (tw, th), _ = cv2.getTextSize(warning, font, font_scale, thickness)
    x = max(10, (W - tw) // 2)
    y = max(th + 20, int(0.12 * H))
    cv2.putText(image, warning, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, warning, (x, y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

    # Subline centered below
    sub_scale = max(0.8, font_scale * 0.6)
    sub_th = max(1, int(sub_scale * 2))
    (stw, sth), _ = cv2.getTextSize(subline, font, sub_scale, sub_th)
    sx = max(10, (W - stw) // 2)
    sy = min(H - 10, y + int(th * 1.5))
    cv2.putText(image, subline, (sx, sy), font, sub_scale, (0, 0, 0), sub_th + 2, cv2.LINE_AA)
    cv2.putText(image, subline, (sx, sy), font, sub_scale, (0, 0, 255), sub_th, cv2.LINE_AA)

    return image
