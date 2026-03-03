#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from hard_windows import HardWindow, load_hard_windows
from screen_utils import get_screen_size, overlay_screen_warning


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Step through hard windows listed in a windows file. '
            'Each frame waits for a keypress so you can inspect the mined segments before annotation.'
        )
    )
    p.add_argument('windows', type=Path, help='Path to the hard windows .txt file.')
    p.add_argument('--max-side', type=int, default=1400, help='Resize frames to fit this maximum side for display.')
    return p.parse_args()


def _resize_to_max_side(frame, max_side: int):
    h, w = frame.shape[:2]
    scale = min(1.0, float(max(1, int(max_side))) / float(max(h, w, 1)))
    if abs(scale - 1.0) < 1e-6:
        return frame
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _draw_overlay(frame, *, window: HardWindow, window_idx: int, total_windows: int, frame_idx: int) -> None:
    text1 = (
        f'window {window_idx + 1}/{total_windows}  '
        f'frames {window.start_frame}-{window.end_frame}  peak={window.peak_frame}  score={window.score:.3f}'
    )
    text2 = f'{window.video_path.name}  frame {frame_idx}'
    text3 = 'Space/Enter next frame, n next window, p previous window, Esc quit'
    cv2.putText(frame, text1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if window.notes:
        cv2.putText(frame, window.notes[:110], (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        text3,
        (12, max(28, frame.shape[0] - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _show_window(win_name: str, window: HardWindow, *, max_side: int, window_idx: int, total_windows: int) -> str:
    cap = cv2.VideoCapture(str(window.video_path))
    if not cap.isOpened():
        return 'next'

    frame_idx = int(window.start_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    screen = get_screen_size()

    while frame_idx <= int(window.end_frame):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        canvas = _resize_to_max_side(frame, int(max_side)).copy()
        _draw_overlay(canvas, window=window, window_idx=window_idx, total_windows=total_windows, frame_idx=frame_idx)
        canvas = overlay_screen_warning(canvas, screen)
        cv2.imshow(win_name, canvas)

        key = cv2.waitKey(0)
        if key == 27:
            cap.release()
            return 'quit'
        if key in (ord('p'), ord('P')):
            cap.release()
            return 'prev'
        if key in (ord('n'), ord('N')):
            cap.release()
            return 'next'
        if key in (13, 32):
            frame_idx += 1
            continue
        frame_idx += 1

    cap.release()
    return 'next'


def main() -> int:
    args = _parse_args()
    windows = load_hard_windows(Path(args.windows))
    if not windows:
        raise SystemExit(f'No windows found in: {args.windows}')

    win_name = 'hard-windows'
    cv2.namedWindow(win_name)
    idx = 0
    while 0 <= idx < len(windows):
        action = _show_window(
            win_name,
            windows[idx],
            max_side=int(args.max_side),
            window_idx=idx,
            total_windows=len(windows),
        )
        if action == 'quit':
            break
        if action == 'prev':
            idx = max(0, idx - 1)
            continue
        idx += 1

    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
