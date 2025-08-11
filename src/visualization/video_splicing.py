"""
video_splicing.py – “always-centred” highlight-clip generator
-------------------------------------------------------------

Each output clip is a fixed-size MP4 (OUTPUT_WIDTH × OUTPUT_HEIGHT) in which the
tracked subject’s bounding-box height always occupies
TARGET_BBOX_HEIGHT_RATIO of the frame height, with optional EMA smoothing on the
zoom factor.

Requires:
    * OpenCV (`cv2`)
    * NumPy
    * tqdm
    * your existing `common_types`, `settings`, and `video_io` modules
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from common_types import BoundingBox, Detection, Track, TrackId
from settings import (
    VIDEO_SUFFIX_SECONDS,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    TARGET_BBOX_HEIGHT_RATIO,
    SMOOTHING_ALPHA,
    MIN_SCALE,
    MAX_SCALE,
)
from util.video_io import VideoReader, VideoWriter, get_video_properties

# --------------------------------------------------------------------------- #
# Helper – per-frame zoom + centre                                            #
# --------------------------------------------------------------------------- #


def _scale_and_center(
    frame: np.ndarray,
    bbox: BoundingBox,
    output_size: Tuple[int, int],
    target_ratio: float,
    prev_scale: float | None = None,
    smoothing_alpha: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Resize the *whole* frame so that bbox.height == target_ratio * output_height,
    then crop a fixed window centred on the (scaled) bbox centre.

    Returns
    -------
    output_frame  Fixed-size BGR image (output_size) with black padding as needed.
    used_scale    The scale factor that was applied (after smoothing / clipping).
    """
    out_w, out_h = output_size

    # instantaneous scale to make bbox fill `target_ratio` of output height
    s_inst = (target_ratio * out_h) / bbox.height
    s_inst = float(np.clip(s_inst, MIN_SCALE, MAX_SCALE))

    # optional exponential smoothing
    s = (
        smoothing_alpha * prev_scale + (1 - smoothing_alpha) * s_inst
        if prev_scale is not None and 0.0 < smoothing_alpha < 1.0
        else s_inst
    )

    # resize whole frame
    scaled_h = int(frame.shape[0] * s)
    scaled_w = int(frame.shape[1] * s)
    scaled = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # new centre of bbox in the scaled frame
    cx, cy = bbox.center
    scx = int(cx * s)
    scy = int(cy * s)

    # crop window [x1:x2, y1:y2] around (scx, scy)
    x1 = scx - out_w // 2
    y1 = scy - out_h // 2
    x2 = x1 + out_w
    y2 = y1 + out_h

    output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # intersection between crop and scaled frame
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(scaled_w, x2), min(scaled_h, y2)
    ox1, oy1 = sx1 - x1, sy1 - y1
    ox2, oy2 = ox1 + (sx2 - sx1), oy1 + (sy2 - sy1)

    if sx2 > sx1 and sy2 > sy1:
        output[oy1:oy2, ox1:ox2] = scaled[sy1:sy2, sx1:sx2]

    return output, s


# --------------------------------------------------------------------------- #
# Utility                                                                     #
# --------------------------------------------------------------------------- #


def _find_detection_at_frame(track_data: List[Detection], frame_idx: int) -> Detection | None:
    """Return the Detection whose `frame_idx` equals *frame_idx*, or None."""
    for d in track_data:
        if d.frame_idx == frame_idx:
            return d
    return None


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def generate_individual_videos(
    tracks: List[Track],
    original_video_path: os.PathLike | str,
    output_dir: os.PathLike | str,
    video_suffix_seconds: float = VIDEO_SUFFIX_SECONDS,
) -> List[Path]:
    """
    Write one centred, scale-normalised MP4 for each Track in *tracks*.

    Each output clip has resolution (OUTPUT_WIDTH × OUTPUT_HEIGHT).
    """
    logger = logging.getLogger(__name__)

    if not tracks:
        logger.warning('No tracks supplied – nothing to do.')
        return []

    os.makedirs(output_dir, exist_ok=True)
    input_name = Path(original_video_path).stem

    props = get_video_properties(original_video_path)
    total_frames = props.total_frames

    logger.info(
        'Generating %d videos (%d×%d px, target ratio %.2f)…',
        len(tracks),
        OUTPUT_WIDTH,
        OUTPUT_HEIGHT,
        TARGET_BBOX_HEIGHT_RATIO,
    )

    # prepare one writer and one scale history per track
    writers: Dict[TrackId, VideoWriter] = {}
    prev_scales: Dict[TrackId, float | None] = {}

    for track in tracks:
        out_path = Path(output_dir) / f'{input_name}+{track.track_id:02d}.mp4'
        vw = VideoWriter(out_path, OUTPUT_WIDTH, OUTPUT_HEIGHT, props.fps)
        vw.start_writing()
        writers[track.track_id] = vw
        prev_scales[track.track_id] = None

        # optional: JSON with first-appearance timestamp
        (Path(output_dir) / f'{input_name}+{track.track_id:02d}.start_time.json').write_text(
            json.dumps({'start_time': track.start_frame() / props.fps})
        )

    # iterate through original video
    with VideoReader(original_video_path) as rdr:
        for frame_idx, frame in tqdm(rdr.read_frames(), total=total_frames, desc='Writing clips'):
            for track in tracks:
                det = _find_detection_at_frame(track.sorted_detections, frame_idx)

                # keep last bbox for a few seconds after the track ends
                if (
                    det is None
                    and frame_idx <= track.end_frame() + int(props.fps * video_suffix_seconds)
                    and track.sorted_detections
                ):
                    det = track.sorted_detections[-1]

                if det is None:
                    continue

                centred, used_scale = _scale_and_center(
                    frame,
                    det.bbox,
                    (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                    TARGET_BBOX_HEIGHT_RATIO,
                    prev_scale=prev_scales[track.track_id],
                    smoothing_alpha=SMOOTHING_ALPHA,
                )
                prev_scales[track.track_id] = used_scale
                writers[track.track_id].write_frame(centred)

    # finalise files
    for w in writers.values():
        w.finish_writing()

    logger.info('✔  Saved %d videos to %s', len(writers), output_dir)
    return [Path(w.output_path) for w in writers.values()]
