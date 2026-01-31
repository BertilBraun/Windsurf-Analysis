import os
import cv2
import numpy as np
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from vidstab import VidStab

from ..util.cache import cache_to_file
from ..util.video_io import VideoReader
from ..motion.gmc import GMC

from ..util.video_io import get_video_properties, get_video_total_frame_count

Transform = NamedTuple('Transform', [('dx', float), ('dy', float), ('da', float), ('frame_idx', int)])
# NOTE:
# - For *raw motion estimation*, `Transform(dx,dy,da,frame_idx=k)` represents the estimated prev->curr delta
#   between frames k-1 and k.
# - For *stabilization rendering*, some call sites use per-frame *absolute correction/warp* values anchored at
#   `frame_idx=k` to apply directly when drawing frame k.


STABLE_GFTT_MAX_CORNERS = 200
STABLE_GFTT_QUALITY_LEVEL = 0.05
STABLE_GFTT_MIN_DISTANCE = 30.0
STABLE_GFTT_BLOCK_SIZE = 3
STABLE_SMOOTHING_WINDOW = 10


def stable_processing_max_dim_half(input_video: str | os.PathLike) -> int:
    props = get_video_properties(input_video)
    return max(1, int(round(max(int(props.width), int(props.height)) / 2.0)))


class MaskedVidStabEstimator:
    """
    VidStab-like motion estimator, but supports per-frame masks.

    This follows the same high-level approach as `vidstab.VidStab`:
    - Detect GFTT keypoints on prev frame (optionally masked)
    - Track with LK optical flow to current frame
    - Estimate a partial affine transform (prev -> curr)

    Returns per-frame deltas mapping points from prev -> curr:
        p_curr = R(da) * p_prev + [dx, dy]
    """

    def __init__(
        self,
        *,
        processing_max_dim: int | float = float('inf'),
        max_corners: int = STABLE_GFTT_MAX_CORNERS,
        quality_level: float = STABLE_GFTT_QUALITY_LEVEL,
        min_distance: float = STABLE_GFTT_MIN_DISTANCE,
        block_size: int = STABLE_GFTT_BLOCK_SIZE,
    ) -> None:
        self.processing_max_dim = float(processing_max_dim)
        self.feature_params: dict[str, object] = dict(
            maxCorners=int(max_corners),
            qualityLevel=float(quality_level),
            minDistance=float(min_distance),
            blockSize=int(block_size),
        )

        self._initialized = False
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None  # (N,1,2) float32 in processed resolution

    def _resize_gray(self, gray: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = gray.shape[:2]
        max_dim = max(h, w)
        if max_dim <= 0:
            return gray, 1.0
        if max_dim <= self.processing_max_dim:
            return gray, 1.0
        scale = float(self.processing_max_dim) / float(max_dim)
        out = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        return out, scale

    def _detect_points(self, gray_proc: np.ndarray, mask_proc: np.ndarray | None) -> np.ndarray:
        if mask_proc is not None and mask_proc.dtype != np.uint8:
            mask_proc = mask_proc.astype(np.uint8, copy=False)
        pts = cv2.goodFeaturesToTrack(gray_proc, mask=mask_proc, **self.feature_params)
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32, copy=False)

    def apply(
        self,
        *,
        frame_idx: int,
        frame_bgr: np.ndarray,
        excluded_bboxes: Sequence[Sequence[int]] | None = None,
        mask_margin_px: int = 0,
    ) -> Transform | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_proc, scale = self._resize_gray(gray)

        mask_proc = None
        if excluded_bboxes:
            mask_full = build_keypoint_mask(
                frame_shape=frame_bgr.shape,
                excluded_bboxes=excluded_bboxes,
                margin_px=mask_margin_px,
            )
            if scale != 1.0:
                h2, w2 = gray_proc.shape[:2]
                mask_proc = cv2.resize(mask_full, (w2, h2), interpolation=cv2.INTER_NEAREST)
            else:
                mask_proc = mask_full

        cur_pts = self._detect_points(gray_proc, mask_proc)

        if not self._initialized:
            self._initialized = True
            self._prev_gray = gray_proc.copy()
            self._prev_pts = cur_pts.copy()
            return None

        assert self._prev_gray is not None
        assert self._prev_pts is not None

        if self._prev_pts.size == 0:
            self._prev_gray = gray_proc.copy()
            self._prev_pts = cur_pts.copy()
            return Transform(dx=0.0, dy=0.0, da=0.0, frame_idx=int(frame_idx))

        matched, status, _err = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray_proc, self._prev_pts, None)
        if matched is None or status is None:
            self._prev_gray = gray_proc.copy()
            self._prev_pts = cur_pts.copy()
            return Transform(dx=0.0, dy=0.0, da=0.0, frame_idx=int(frame_idx))

        prev_good = self._prev_pts[status.flatten() == 1]
        curr_good = matched[status.flatten() == 1]

        H = None
        if prev_good.shape[0] >= 4 and curr_good.shape[0] == prev_good.shape[0]:
            H, _inliers = cv2.estimateAffinePartial2D(prev_good, curr_good)

        if H is None:
            dx = dy = da = 0.0
        else:
            dx = float(H[0, 2])
            dy = float(H[1, 2])
            da = float(np.arctan2(float(H[1, 0]), float(H[0, 0])))

            # Map translation back to original resolution.
            if scale != 1.0:
                dx /= scale
                dy /= scale

        self._prev_gray = gray_proc.copy()
        self._prev_pts = cur_pts.copy()

        return Transform(dx=dx, dy=dy, da=da, frame_idx=int(frame_idx))


def compute_stabilization_transforms_masked_vidstab(
    input_video: str | os.PathLike,
    *,
    bboxes_by_frame: Mapping[int, Sequence[Sequence[int]]] | None = None,
    mask_margin_px: int = 20,
    processing_max_dim: int | None = None,
    max_corners: int = STABLE_GFTT_MAX_CORNERS,
    quality_level: float = STABLE_GFTT_QUALITY_LEVEL,
    min_distance: float = STABLE_GFTT_MIN_DISTANCE,
    block_size: int = STABLE_GFTT_BLOCK_SIZE,
    limit_frames: int | None = None,
) -> list[Transform]:
    """
    Compute per-frame camera motion deltas (prev->curr) using a VidStab-like GFTT+LK estimator with optional bbox masks.

    Returns `Transform` entries for frames 1..N-1 (frame 0 has no prev frame, so returns None and is omitted).
    """
    if processing_max_dim is None:
        processing_max_dim = stable_processing_max_dim_half(input_video)

    estimator = MaskedVidStabEstimator(
        processing_max_dim=int(processing_max_dim),
        max_corners=int(max_corners),
        quality_level=float(quality_level),
        min_distance=float(min_distance),
        block_size=int(block_size),
    )
    out: list[Transform] = []
    with VideoReader(input_video) as reader:
        for frame_idx, frame in reader.read_frames():
            frame_idx = int(frame_idx)
            if limit_frames is not None and frame_idx >= int(limit_frames):
                break
            excluded = None if bboxes_by_frame is None else bboxes_by_frame.get(frame_idx, ())
            t = estimator.apply(
                frame_idx=frame_idx,
                frame_bgr=frame,
                excluded_bboxes=excluded,
                mask_margin_px=int(mask_margin_px),
            )
            if t is not None:
                out.append(t)
    return out


@cache_to_file('vidstab_transforms')
def compute_stabilization_transforms(input_video: str | os.PathLike) -> list[Transform]:
    """
    Compute per-frame stabilization deltas using VidStab's generated transforms.
    Output is dx, dy, da for each frame relative to the previous frame (no cumulative, no prepend).
    """
    total_frames = get_video_total_frame_count(input_video)

    stabilizer = VidStab()  # TODO: 'DENSE')
    stabilizer.gen_transforms(
        input_path=input_video, smoothing_window=min(20, int(total_frames) - 1)
    )

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
    margin_px: int = 0,
) -> np.ndarray:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    mask = np.full((height, width), 255, dtype=np.uint8)

    margin = max(0, int(margin_px))
    for bbox in excluded_bboxes:
        if len(bbox) != 4:
            raise ValueError(f'Invalid bbox: {bbox}')
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        x1 = max(0, min(width, x1 - margin))
        y1 = max(0, min(height, y1 - margin))
        x2 = max(0, min(width, x2 + margin))
        y2 = max(0, min(height, y2 + margin))
        if x2 <= x1 or y2 <= y1:
            continue
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


def vidstab_like_correction_by_frame(
    raw_motion_transforms: list[Transform],
    *,
    frame_count: int,
    smoothing_window: int = 30,
) -> list[Transform]:
    """
    Compute per-frame *absolute* stabilization corrections suitable for direct rendering.

    The input `raw_motion_transforms` are prev->curr motion deltas for frames 1..N-1 (frame 0 has no delta).

    Returns a dense list of length `frame_count` with:
      - frame 0: identity (0,0,0)
      - frame k (k>=1): correction[k] = smoothed_trajectory[k] - trajectory[k]
        where trajectories are cumulative sums of raw motion deltas.

    This matches the convention used by the Qt/web players (apply correction directly to frame k when drawing).
    """
    frame_count = int(frame_count)
    if frame_count <= 0:
        return []

    if frame_count == 1:
        return [Transform(dx=0.0, dy=0.0, da=0.0, frame_idx=0)]

    raw_by_frame: dict[int, Transform] = {int(t.frame_idx): t for t in raw_motion_transforms}

    raw_deltas: list[list[float]] = []
    for frame_idx in range(1, frame_count):
        t = raw_by_frame.get(int(frame_idx))
        if t is None:
            raw_deltas.append([0.0, 0.0, 0.0])
        else:
            raw_deltas.append([float(t.dx), float(t.dy), float(t.da)])

    raw = np.asarray(raw_deltas, dtype=np.float64)  # shape (frame_count-1,3)
    if raw.shape[0] == 0:
        return [Transform(dx=0.0, dy=0.0, da=0.0, frame_idx=i) for i in range(frame_count)]

    traj = np.cumsum(raw, axis=0)  # shape (frame_count-1,3)
    n = max(1, min(int(smoothing_window), int(traj.shape[0])))
    traj_s = _bfill_rolling_mean(traj, n=n)  # shape (frame_count-1,3)
    corr = traj_s - traj  # shape (frame_count-1,3)

    out: list[Transform] = [Transform(dx=0.0, dy=0.0, da=0.0, frame_idx=0)]
    for frame_idx in range(1, frame_count):
        dx, dy, da = corr[frame_idx - 1]
        out.append(Transform(dx=float(dx), dy=float(dy), da=float(da), frame_idx=int(frame_idx)))
    return out


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
