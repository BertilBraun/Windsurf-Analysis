# Camera motion compensation

import math
import numpy as np

NDArrayF = np.ndarray

from server.inference.src.common_types import FrameIndex
from server.inference.src.visualization.stabilize import Transform


class CMC:
    """Camera motion compensation. Functionality to apply camera motion compensation Transforms to a KF state."""

    def __init__(self, transforms: list[Transform]):
        self._transforms_dict: dict[FrameIndex, Transform] = {t.frame_idx - 1: t for t in transforms}

    def apply_forward(self, mean: NDArrayF, cov: NDArrayF, frame_idx: FrameIndex) -> tuple[NDArrayF, NDArrayF]:
        """Apply one prev→curr delta to KF state and covariance. Advances from frame_idx to frame_idx+1."""
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
