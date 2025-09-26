# kalman_tracking.py
# Typed, minimal Kalman filter for bbox tracking.
# Fixes: stable covariance (Joseph form), controlled Q growth, shrink after update,
# inflation only on misses, optional inflation caps, vectorized multi_predict.

from __future__ import annotations

import math
from typing import Dict, Literal, Tuple
import numpy as np
import numpy.typing as npt
import scipy.linalg

from server.inference.src.visualization.stabilize import Transform

NDArrayF = npt.NDArray[np.float64]

CHI2_QUANTILES: Dict[float, Dict[int, float]] = {
    0.0: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
    0.90: {1: 2.7055, 2: 4.6052, 3: 6.2514, 4: 7.779},
    0.95: {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877},
}


class KalmanFilter:
    """
    State: [cx, cy, w, h, vx, vy, vw, vh]. Constant velocity. Measure [cx, cy, w, h].
    cx, cy: center x, y of the bounding box
    w, h: width, height of the bounding box
    vx, vy: velocity x, y of the bounding box
    vw, vh: velocity width, height of the bounding box

    Notes to avoid runaway growth:
      - Use small but >0 measurement noise. Default is conservative.
      - Use Joseph update to keep covariance PSD and shrinking after updates.
      - Interpret missed_frames as the full gap LENGTH advanced in this call.
        Do NOT also loop per-frame with missed_frames>0, or growth compounds.
      - Inflate display bbox only when missed_count>0.
      - Optional caps prevent extreme inflation.
    """

    def __init__(
        self,
        dt: float = 1.0,
        proc_std_weight_pos: float = 1.0 / 20.0,
        proc_std_weight_vel: float = 1.0 / 80.0,
        meas_std_weight_pos: float = 1.0 / 50.0,
        meas_std_weight_size: float = 1.0 / 20.0,
        q_growth: float = 1.05,
    ) -> None:
        self.ndim = 4
        self.dt = float(dt)

        self._F_base: NDArrayF = np.eye(2 * self.ndim, dtype=np.float64)
        for i in range(self.ndim):
            self._F_base[i, self.ndim + i] = self.dt

        self._H: NDArrayF = np.eye(self.ndim, 2 * self.ndim, dtype=np.float64)

        self._proc_std_weight_pos = float(proc_std_weight_pos)
        self._proc_std_weight_vel = float(proc_std_weight_vel)
        self._meas_std_weight_pos = float(meas_std_weight_pos)
        self._meas_std_weight_size = float(meas_std_weight_size)
        self._q_growth = float(q_growth)

    # ---------------------------- Utilities ---------------------------- #

    def _F_gap(self, gap: int) -> NDArrayF:
        """State transition for an integer gap of frames."""
        F = self._F_base.copy()
        if gap != 1:
            for i in range(self.ndim):
                F[i, self.ndim + i] = self.dt * gap
        return F

    def _meas_std(self, mean: NDArrayF) -> NDArrayF:
        """Measurement stds for [cx,cy,w,h], proportional to w/h."""
        return np.array(
            [
                self._meas_std_weight_pos * max(mean[2], 1e-6),
                self._meas_std_weight_pos * max(mean[3], 1e-6),
                self._meas_std_weight_size * max(mean[2], 1e-6),
                self._meas_std_weight_size * max(mean[3], 1e-6),
            ],
            dtype=np.float64,
        )

    # ---------------------------- Single-track ---------------------------- #

    def initiate(self, measurement: NDArrayF) -> Tuple[NDArrayF, NDArrayF]:
        mean_pos = measurement.astype(np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = np.array(
            [
                2 * self._proc_std_weight_pos * measurement[2],  # x
                2 * self._proc_std_weight_pos * measurement[3],  # y
                2 * self._proc_std_weight_pos * measurement[2],  # w
                2 * self._proc_std_weight_pos * measurement[3],  # h
                10 * self._proc_std_weight_vel * measurement[2],  # vx
                10 * self._proc_std_weight_vel * measurement[3],  # vy
                10 * self._proc_std_weight_vel * measurement[2],  # vw
                10 * self._proc_std_weight_vel * measurement[3],  # vh
            ],
            dtype=np.float64,
        )
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self,
        mean: NDArrayF,
        covariance: NDArrayF,
        missed_frames: int = 0,
    ) -> Tuple[NDArrayF, NDArrayF]:
        """
        One-step prediction.

        Args:
            mean: (8,)
            covariance: (8,8)
            missed_frames: number of consecutive frames without an update;
                           scales Q by (_q_growth ** missed_frames) to accelerate spread.

        Returns:
            predicted mean, covariance
        """
        gap = int(max(missed_frames, 0)) if missed_frames else 1
        F = self._F_gap(gap)

        std_pos = np.array(
            [
                self._proc_std_weight_pos * max(mean[2], 1e-6),
                self._proc_std_weight_pos * max(mean[3], 1e-6),
                self._proc_std_weight_pos * max(mean[2], 1e-6),
                self._proc_std_weight_pos * max(mean[3], 1e-6),
            ],
            dtype=np.float64,
        )
        std_vel = np.array(
            [
                self._proc_std_weight_vel * max(mean[2], 1e-6),
                self._proc_std_weight_vel * max(mean[3], 1e-6),
                self._proc_std_weight_vel * max(mean[2], 1e-6),
                self._proc_std_weight_vel * max(mean[3], 1e-6),
            ],
            dtype=np.float64,
        )

        # Scale Q once for the whole gap to avoid compounding errors.
        q_scale = self._q_growth**gap
        Q = np.diag(np.square(np.r_[std_pos, std_vel])) * q_scale

        mean = mean @ F.T
        P = F @ covariance @ F.T + Q
        # Symmetrize to avoid drift from numerical noise.
        P = 0.5 * (P + P.T)
        return mean, P

    def project(self, mean: NDArrayF, covariance: NDArrayF) -> Tuple[NDArrayF, NDArrayF, NDArrayF]:
        z_mean = self._H @ mean
        S_wo_R = self._H @ covariance @ self._H.T
        R_std = self._meas_std(mean)
        R = np.diag(np.square(R_std))
        S = S_wo_R + R
        return z_mean, S, R

    def update(self, mean: NDArrayF, covariance: NDArrayF, measurement: NDArrayF) -> Tuple[NDArrayF, NDArrayF]:
        """
        Measurement update with z = [cx, cy, w, h].

        Returns:
            updated mean, covariance
        """
        z_pred, S, R = self.project(mean, covariance)
        chol, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve((chol, lower), (covariance @ self._H.T).T, check_finite=False).T
        y = measurement - z_pred

        I = np.eye(covariance.shape[0], dtype=np.float64)  # noqa: E741 (I is the identity matrix)
        KH = K @ self._H
        mean_new = mean + K @ y
        P_bar = (I - KH) @ covariance
        P_new = P_bar @ (I - KH).T + K @ R @ K.T
        P_new = 0.5 * (P_new + P_new.T)
        return mean_new, P_new

    def gating_distance(
        self,
        mean: NDArrayF,
        covariance: NDArrayF,
        measurements: NDArrayF,
        only_position: bool = True,
        metric: Literal['maha', 'gaussian'] = 'maha',
    ) -> NDArrayF:
        """
        Squared distance between state distribution and candidate detections.

        Args:
            mean: (8,)
            covariance: (8,8)
            measurements: (N,4) with [cx, cy, w, h]
            only_position: if True, use [cx,cy] only and 2x2 S
            metric: "maha" for Mahalanobis, "gaussian" for plain squared Euclidean

        Returns:
            (N,) distances
        """
        z_pred, S, _ = self.project(mean, covariance)
        if only_position:
            z_pred = z_pred[:2]
            S = S[:2, :2]
            Z = measurements[:, :2]
        else:
            Z = measurements

        d = Z - z_pred  # (N,m)

        if metric == 'gaussian':
            return np.sum(d * d, axis=1)

        # Mahalanobis: d^T S^{-1} d using Cholesky of S.
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            # Add tiny jitter if numerical issues.
            jitter = 1e-9 * np.eye(S.shape[0], dtype=np.float64)
            L = np.linalg.cholesky(S + jitter)

        z = scipy.linalg.solve_triangular(L, d.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)

    def display_bbox(
        self,
        mean: NDArrayF,
        covariance: NDArrayF,
        alpha: float = 0.90,
        include_size_unc: bool = False,
    ) -> NDArrayF:
        """
        Inflate around the current box using projected covariance.
        If base_from_measurement=True, base w/h come from current mean (normal).
        Returns [cx, cy, w_out, h_out].
        """
        z_mean, S, _ = self.project(mean, covariance)

        cx, cy, w, h = map(float, z_mean)
        k = np.sqrt(CHI2_QUANTILES[alpha][2])

        dx = k * np.sqrt(max(S[0, 0], 0.0))
        dy = k * np.sqrt(max(S[1, 1], 0.0))

        w_out = w + 2.0 * dx
        h_out = h + 2.0 * dy

        if include_size_unc:
            k_size = np.sqrt(CHI2_QUANTILES[alpha][1])
            w_out += k_size * np.sqrt(max(S[2, 2], 0.0))
            h_out += k_size * np.sqrt(max(S[3, 3], 0.0))

        return np.array([cx, cy, max(w_out, 1e-3), max(h_out, 1e-3)], dtype=np.float64)

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

    def apply_forward_gmc_state(self, mean: NDArrayF, cov: NDArrayF, transform: Transform) -> tuple[NDArrayF, NDArrayF]:
        """Apply one prev→curr delta to KF state and covariance."""
        A, T = self._build_A_T(transform)
        m = A @ mean
        m[0:2] += T
        P = A @ cov @ A.T
        return m, P

    def advance_state_to_frame(
        self,
        mean: NDArrayF,
        cov: NDArrayF,
        transforms: Dict[int, Transform],
        from_frame: int,
        to_frame: int,
    ) -> tuple[NDArrayF, NDArrayF]:
        """Step from `from_frame` to `to_frame` using per-frame forward deltas.
        Assumes `frame_warp[f]` is the delta (f-1 -> f)."""
        if to_frame <= from_frame:
            return mean, cov
        m, P = mean, cov
        # hop f = from_frame+1 .. to_frame
        for f in range(from_frame + 1, to_frame + 1):
            m, P = self.predict(m, P, missed_frames=1)
            m, P = self.apply_forward_gmc_state(m, P, transforms[f])
        return m, P
