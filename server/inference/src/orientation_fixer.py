#!/usr/bin/env python3
"""
orientation_fixer.py

Loads a YOLO classification model once, then for each input video:
  - samples one batch of frames (uniform or random),
  - runs a single forward pass (one batch),
  - majority-votes the predicted 0/90/180/270 class,
  - rotates the entire video with FFmpeg to upright (0°),
  - writes the corrected video.

Dependencies:
  pip install ultralytics opencv-python
  (and make sure `ffmpeg` is on PATH)
"""

import random
import shutil
import subprocess
from pathlib import Path
from collections import Counter
from typing import Literal

import cv2
import numpy as np

from ultralytics import YOLO

from server.inference.src.util.video_io import get_video_properties

DEGREES = (0, 90, 180, 270)


# -------------------
# OrientationFixer
# -------------------
class OrientationFixer:
    def __init__(self, model_path: str):
        """
        model_path: path to your trained classification weights (best.pt)
        """
        self.model = YOLO(model_path)  # loads once

        # Build class index -> degree mapping from names (expects labels 0/90/180/270)
        names = getattr(self.model, 'names', None)
        if names is None:
            # assume 0,1,2,3 => 0,90,180,270
            self.idx2deg = {0: 0, 1: 90, 2: 180, 3: 270}
        else:
            # names may be dict or list; normalize to {idx: "name"}
            if isinstance(names, dict):
                items = names.items()
            else:
                items = enumerate(names)
            m = {}
            for k, v in items:
                label = str(v).strip().lower().replace('deg', '').replace('°', '')
                if label in {'0', '90', '180', '270'}:
                    m[int(k)] = int(label)
            if len(m) != 4:
                raise ValueError(f'Expected class names 0/90/180/270, got: {names}')
            self.idx2deg = m

    def detect_orientation(
        self,
        input_video: str,
        n_samples: int = 16,
        sampling: Literal['uniform', 'random'] = 'uniform',
    ) -> int:
        """
        Returns the dominant orientation of the video (0, 90, 180, 270)
        """
        input_video = str(input_video)

        props = get_video_properties(input_video)

        # Pick indices and read frames (one batch per video)
        indices = _sample_frame_indices(props.total_frames, n_samples, sampling)
        frames = _read_frames_at_indices(input_video, indices)

        if not frames:
            print(f'Warning: Could not read any frames from {input_video}')
            return 0

        # One forward pass (single batch)
        results = self.model(frames, verbose=False)
        predicted_deg = [self.idx2deg[int(r.probs.top1)] for r in results]

        return _majority_vote(predicted_deg)

    def fix_video(self, input_video: str, output_video: str | None = None) -> tuple[str, int]:
        """
        Rotates the video so that the dominant orientation is upright (0°).
        """
        if output_video is None:
            p = Path(input_video)
            output_video = str(p.with_name(p.stem + '_upright' + p.suffix))

        dominant = self.detect_orientation(input_video)
        print(f'Dominant orientation: {dominant}')
        _apply_rotation_ffmpeg(input_video, output_video, dominant)
        return output_video, dominant


# -------------------
# FFmpeg helpers
# -------------------
def _rotation_filter(deg: int) -> str:
    """
    Return FFmpeg -vf filter to rotate by 'deg' clockwise (0/90/180/270).
    """
    d = deg % 360
    if d == 0:
        return ''
    if d == 90:
        return 'transpose=1'  # 90° CW
    if d == 180:
        return 'transpose=1,transpose=1'  # 180°
    if d == 270:
        return 'transpose=2'  # 90° CCW
    raise ValueError('deg must be one of {0,90,180,270}')


def _apply_rotation_ffmpeg(in_path: str, out_path: str, content_is_at_deg: int) -> None:
    """
    Rotate whole video so that predicted content orientation (content_is_at_deg)
    becomes upright (0°). That means applying -deg modulo 360.
    """
    rotate_deg_to_upright = (-content_is_at_deg) % 360
    vf = _rotation_filter(rotate_deg_to_upright)
    if vf:
        cmd = [
            'ffmpeg',
            '-y',
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            in_path,
            '-vf',
            vf,
            '-c:v',
            'libx264',
            '-preset',
            'medium',
            '-crf',
            '18',
            '-c:a',
            'copy',
            out_path,
        ]
        print(f'Applying rotation with FFmpeg: {cmd}')
        subprocess.check_call(cmd)
        print(f'Rotation applied to {in_path} and saved to {out_path}')
    else:
        # No rotation needed; copy the file
        shutil.copy(in_path, out_path)


# -------------------
# Frame sampling
# -------------------
def _sample_frame_indices(total_frames: int, n_samples: int, mode: str) -> list[int]:
    if total_frames <= 0:
        return []
    n = min(n_samples, total_frames)
    if n <= 0:
        return []
    if mode == 'uniform':
        # spread roughly evenly across the whole video
        if n == 1:
            return [total_frames // 2]
        step = total_frames / float(n)
        return [int(round(i * step + step / 2 - 0.5)) for i in range(n)]
    else:  # random
        return sorted(random.sample(range(total_frames), n))


def _read_frames_at_indices(video_path: str, indices: list[int]) -> list[np.ndarray]:
    """
    Read RGB frames at given indices. Returns a list of np.ndarray (H, W, 3) RGB.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _majority_vote(deg_preds: list[int]) -> int:
    if not deg_preds:
        return 0
    c = Counter(deg_preds)
    top = c.most_common()
    if len(top) == 1 or (len(top) > 1 and top[0][1] > top[1][1]):
        return top[0][0]
    # tie-break preference: 0 -> 270 -> 90 -> 180 (arbitrary, stable)
    for pref in (0, 270, 90, 180):
        if c[pref] == top[0][1]:
            return pref
    return top[0][0]
