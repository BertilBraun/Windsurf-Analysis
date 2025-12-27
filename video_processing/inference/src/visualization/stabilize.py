import os
import numpy as np
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from vidstab import VidStab

from ..util.cache import cache_to_file
from ..util.video_io import VideoReader
from ..motion.gmc import GMC

from ..util.video_io import get_video_properties

Transform = NamedTuple(
    'Transform', [('dx', float), ('dy', float), ('da', float), ('frame_idx', int)]
)  # dx, dy, da for each frame relative to the previous frame (frame[i] - frame[i-1]) -> frame[i] = frame[i-1] + dx, dy, da


@cache_to_file('vidstab_transforms')
def compute_stabilization_transforms(input_video: str | os.PathLike) -> list[Transform]:
    """
    Compute per-frame stabilization deltas using VidStab's generated transforms.
    Output is dx, dy, da for each frame relative to the previous frame (no cumulative, no prepend).
    """
    video_properties = get_video_properties(input_video)

    stabilizer = VidStab()  # TODO: 'DENSE')
    stabilizer.gen_transforms(input_path=input_video, smoothing_window=min(20, video_properties.total_frames - 1))

    assert stabilizer._raw_transforms is not None
    raw = [
        Transform(float(dx), float(dy), float(da), i)
        for i, (dx, dy, da) in enumerate(stabilizer._raw_transforms, start=1)
    ]
    return raw


def stabilize_video(input_video: str | os.PathLike, output_video: str | os.PathLike, transforms: list[Transform]):
    stabilizer = VidStab()
    stabilizer.transforms = np.array([[transform.dx, transform.dy, transform.da] for transform in transforms])
    stabilizer.apply_transforms(input_video, output_video, show_progress=False)


@cache_to_file('gmc_transforms')
def compute_stabilization_transforms_gmc(
    input_video: str | os.PathLike,
    *,
    bboxes_by_frame: Mapping[int, Sequence[Sequence[int]]] | None = None,
    downscale: int = 2,
) -> list[Transform]:
    """
    Compute per-frame camera motion deltas using the GMC method,
    and return dx, dy, da for each frame relative to the previous frame.
    """
    gmc = GMC(downscale=downscale)
    transforms: list[Transform] = []

    with VideoReader(input_video) as reader:
        for f, frame in reader.read_frames():
            excluded_bboxes = None
            if bboxes_by_frame is not None:
                excluded_bboxes = bboxes_by_frame.get(int(f), ())

            transform = gmc_transform_from_frame(
                gmc,
                frame_idx=int(f),
                frame=frame,
                excluded_bboxes=excluded_bboxes,
            )
            if transform is not None:
                transforms.append(transform)

    return transforms


def gmc_transform_from_frame(
    gmc: GMC,
    *,
    frame_idx: int,
    frame: np.ndarray,
    excluded_bboxes: Sequence[Sequence[int]] | None = None,
) -> Transform | None:
    mask = None
    if excluded_bboxes:
        mask = build_keypoint_mask(frame_shape=frame.shape, excluded_bboxes=excluded_bboxes)

    H_delta = gmc.apply(frame, mask=mask)

    if int(frame_idx) == 0:
        return None

    # Extract rigid delta parameters from H (prev -> curr)
    R00, tx = float(H_delta[0, 0]), float(H_delta[0, 2])
    R10, ty = float(H_delta[1, 0]), float(H_delta[1, 2])
    return Transform(
        dx=float(tx),
        dy=float(ty),
        da=float(np.arctan2(R10, R00)),
        frame_idx=int(frame_idx),
    )


def build_keypoint_mask(
    *,
    frame_shape: tuple[int, int, int] | tuple[int, int],
    excluded_bboxes: Sequence[Sequence[int]],
) -> np.ndarray:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    mask = np.full((height, width), 255, dtype=np.uint8)

    for bbox in excluded_bboxes:
        if len(bbox) != 4:
            raise ValueError(f'Invalid bbox: {bbox}')
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        mask[y1:y2, x1:x2] = 0

    return mask


def vidstab_like_transforms(transforms: list[Transform], smoothing_window: int = 30) -> list[Transform]:
    """
    Return per-frame stabilized transforms with VidStab formula:
    transforms_stab[i] = raw[i] + (smoothed_trajectory[i] - trajectory[i])
    """
    # raw per-frame deltas
    raw = np.array([[t.dx, t.dy, t.da] for t in transforms], dtype=np.float64)  # shape (N,3)
    N = raw.shape[0]
    if N == 0:
        return []

    # cumulative trajectory (world-axis sum per column)
    traj = np.cumsum(raw, axis=0)  # shape (N,3)

    # rolling-mean smoothing with VidStab's backfill behavior
    traj_s = _bfill_rolling_mean(traj, n=smoothing_window)  # shape (N,3)

    # VidStab formula
    stab = raw + (traj_s - traj)  # shape (N,3)
    frame_idxs = [t.frame_idx for t in transforms]
    return [
        Transform(dx=float(dx), dy=float(dy), da=float(da), frame_idx=int(frame_idx))
        for frame_idx, (dx, dy, da) in zip(frame_idxs, stab)
    ]


def _bfill_rolling_mean(arr: np.ndarray, n: int = 30) -> np.ndarray:
    if arr.shape[0] < n:
        raise ValueError('arr.shape[0] cannot be less than n')
    if n == 1:
        return arr
    pre_buffer = np.zeros(3, dtype=arr.dtype).reshape(1, 3)
    post_buffer = np.zeros(3 * n, dtype=arr.dtype).reshape(n, 3)
    arr_cumsum = np.cumsum(np.vstack((pre_buffer, arr, post_buffer)), axis=0)
    buffer_roll_mean = (arr_cumsum[n:, :] - arr_cumsum[:-n, :]) / float(n)
    trunc_roll_mean = buffer_roll_mean[:-n, :]
    bfill_size = arr.shape[0] - trunc_roll_mean.shape[0]
    bfill = np.tile(trunc_roll_mean[0, :], (bfill_size, 1))
    return np.vstack((bfill, trunc_roll_mean))
