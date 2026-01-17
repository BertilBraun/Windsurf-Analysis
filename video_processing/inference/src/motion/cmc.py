# Camera motion compensation

import math
import warnings
import numpy as np

NDArrayF = np.ndarray

from ..common_types import FrameIndex
from ..visualization.stabilize import Transform


class CMC:
    """Camera motion compensation. Functionality to apply camera motion compensation Transforms to a KF state."""

    def __init__(self, transforms: list[Transform]):
        # Convention:
        # - Each `Transform` represents the rigid motion mapping points from prev -> curr:
        #     p_curr = R(da) * p_prev + [dx, dy]
        # - We store it keyed by prev_frame_idx, so a transform with frame_idx==k maps (k-1) -> k.
        self._transforms_dict: dict[FrameIndex, Transform] = {t.frame_idx - 1: t for t in transforms}
        if transforms and min(t.frame_idx for t in transforms) == 0:
            warnings.warn(
                'CMC received a Transform with frame_idx==0; expected prev->curr deltas with frame_idx starting at 1 '
                '(where frame_idx==k maps k-1 -> k). If your transforms are indexed by prev-frame instead, CMC will be off by one.',
                RuntimeWarning,
                stacklevel=2,
            )

    def apply_forward(self, mean: NDArrayF, cov: NDArrayF, frame_idx: FrameIndex) -> tuple[NDArrayF, NDArrayF]:
        """Apply one prev->curr delta to KF state and covariance. Advances from frame_idx to frame_idx+1."""
        assert frame_idx in self._transforms_dict, f'Frame Index {frame_idx} is required! There is a bug somewhere!'
        transform = self._transforms_dict[frame_idx]
        A, T = self._build_A_T(transform)
        m = A @ mean
        m[0:2] += T
        P = A @ cov @ A.T
        P = 0.5 * (P + P.T)  # symmetrize to avoid numerical drift
        return m, P

    def _build_A_T(self, transform: Transform) -> tuple[NDArrayF, NDArrayF]:
        """Return (A, T) for forward GMC on KF state [cx,cy,w,h,vx,vy,vw,vh]."""
        A = np.eye(8, dtype=np.float64)
        R = np.array(
            [[math.cos(transform.da), -math.sin(transform.da)], [math.sin(transform.da), math.cos(transform.da)]],
            dtype=np.float64,
        )
        A[0:2, 0:2] = R  # positions
        A[4:6, 4:6] = R  # velocities (cx,cy)
        T = np.array([transform.dx, transform.dy], dtype=np.float64)
        return A, T
