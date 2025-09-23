# kalman_tracking.py
# Typed, minimal Kalman filter for bbox tracking.
# Fixes: stable covariance (Joseph form), controlled Q growth, shrink after update,
# inflation only on misses, optional inflation caps, vectorized multi_predict.

from __future__ import annotations

from typing import Dict, Tuple, Optional, Union
import numpy as np
import numpy.typing as npt
import scipy.linalg

NDArrayF = npt.NDArray[np.float64]

CHI2_QUANTILES: Dict[float, Dict[int, float]] = {
    0.90: {1: 2.7055, 2: 4.6052, 3: 6.2514, 4: 7.779},
    0.95: {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877},
}


class KalmanFilter:
    """
    State: [x, y, w, h, vx, vy, vw, vh]. Constant velocity. Measure [x, y, w, h].

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
        meas_std_weight_pos: float = 1.0 / 400.0,
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
        self._q_growth = float(q_growth)
        self._growth_per_miss = float(growth_per_miss)
        self._growth_cap = float(growth_cap)

    # ---------------------------- Utilities ---------------------------- #

    def _F_gap(self, gap: int) -> NDArrayF:
        """State transition for an integer gap of frames."""
        F = self._F_base.copy()
        if gap != 1:
            for i in range(self.ndim):
                F[i, self.ndim + i] = self.dt * gap
        return F

    def _meas_std(self, mean: NDArrayF) -> NDArrayF:
        """Measurement stds for [x,y,w,h], proportional to w/h."""
        return np.array(
            [
                self._meas_std_weight_pos * max(mean[2], 1e-6),
                self._meas_std_weight_pos * max(mean[3], 1e-6),
                self._meas_std_weight_pos * max(mean[2], 1e-6),
                self._meas_std_weight_pos * max(mean[3], 1e-6),
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
        Measurement update with z = [x, y, w, h].

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
        only_position: bool = False,
        metric: str = 'maha',
    ) -> NDArrayF:
        """
        Squared distance between state distribution and candidate detections.

        Args:
            mean: (8,)
            covariance: (8,8)
            measurements: (N,4) with [x, y, w, h]
            only_position: if True, use [x,y] only and 2x2 S
            metric: "maha" for Mahalanobis, "gaussian" for plain squared Euclidean

        Returns:
            (N,) distances
        """
        assert metric in ['maha', 'gaussian']
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

    def multi_predict(
        self,
        mean: NDArrayF,  # (N,8)
        covariance: NDArrayF,  # (N,8,8)
        missed_frames: Optional[Union[int, npt.NDArray[np.int_]]] = 0,
    ) -> Tuple[NDArrayF, NDArrayF]:
        """
        Vectorized predict. missed_frames is the gap LENGTH per track for this call.
        Pass scalar or (N,) ints. Do not also loop per frame with missed_frames>0.
        """
        N = mean.shape[0]
        assert covariance.shape == (N, 8, 8)

        # Build per-track F and Q
        gaps = (np.zeros(N, dtype=int) + (missed_frames if np.isscalar(missed_frames) else missed_frames)).astype(int)
        gaps[gaps < 1] = 1

        # Means
        mean_pred = mean.copy()
        cov_pred = covariance.copy()
        for i in range(N):
            F = self._F_gap(int(gaps[i]))

            w = max(mean[i, 2], 1e-6)
            h = max(mean[i, 3], 1e-6)
            std_pos = np.array(
                [
                    self._proc_std_weight_pos * w,
                    self._proc_std_weight_pos * h,
                    self._proc_std_weight_pos * w,
                    self._proc_std_weight_pos * h,
                ],
                dtype=np.float64,
            )
            std_vel = np.array(
                [
                    self._proc_std_weight_vel * w,
                    self._proc_std_weight_vel * h,
                    self._proc_std_weight_vel * w,
                    self._proc_std_weight_vel * h,
                ],
                dtype=np.float64,
            )
            q_scale = self._q_growth ** int(gaps[i])
            Q = np.diag(np.square(np.r_[std_pos, std_vel])) * q_scale

            mean_pred[i] = mean[i] @ F.T
            cov_pred[i] = F @ covariance[i] @ F.T + Q
            cov_pred[i] = 0.5 * (cov_pred[i] + cov_pred[i].T)

        return mean_pred, cov_pred

    # ------------------------ Display / Inflation ------------------------ #

    def inflated_bbox(
        self,
        mean: NDArrayF,
        covariance: NDArrayF,
        alpha: float = 0.95,
        include_size_unc: bool = True,
    ) -> NDArrayF:
        """
        Inflate around the current box using projected covariance.
        If base_from_measurement=True, base w/h come from current mean (normal).
        Returns [x, y, w_out, h_out].
        """
        z_mean, S, _ = self.project(mean, covariance)

        k_pos = np.sqrt(CHI2_QUANTILES[alpha][2])
        rx = np.sqrt(max(S[0, 0], 0.0)) * k_pos
        ry = np.sqrt(max(S[1, 1], 0.0)) * k_pos

        growth = min(rx, ry) * max(pos_scale, 0.0)

        x, y, w, h = z_mean
        w_base, h_base = float(w), float(h)

        w_out = w_base + 2.0 * growth
        h_out = h_base + 2.0 * growth

        if include_size_unc:
            k_size = np.sqrt(CHI2_QUANTILES[alpha][1])
            w_out += k_size * np.sqrt(max(S[2, 2], 0.0))
            h_out += k_size * np.sqrt(max(S[3, 3], 0.0))

        return np.array([float(x), float(y), max(w_out, 1e-3), max(h_out, 1e-3)], dtype=np.float64)

    def display_bbox(
        self,
        mean: NDArrayF,
        covariance: NDArrayF,
        missed_count: int,
        alpha_seen: float = 0.0,
        alpha_missed: float = 0.90,
        include_size_unc_on_miss: bool = False,
    ) -> NDArrayF:
        """
        Policy: no inflation when a detection was just used (missed_count==0),
        inflate when misses>0. This guarantees shrink after solid updates.
        """
        if missed_count <= 0:
            # alpha_seen=0 => no inflation. Use raw mean box.
            if alpha_seen <= 0:
                z, _, _ = self.project(mean, covariance)
                x, y, w, h = z
                return np.array([float(x), float(y), float(max(w, 1e-3)), float(max(h, 1e-3))], dtype=np.float64)
            return self.inflated_bbox(mean, covariance, alpha=alpha_seen, include_size_unc=False)
        else:
            # Scale positional inflation gently with sqrt of misses
            pos_scale = min(1.0 + self._growth_per_miss * np.sqrt(float(max(missed_count, 0))), self._growth_cap)
            return self.inflated_bbox(
                mean,
                covariance,
                alpha=alpha_missed,
                include_size_unc=include_size_unc_on_miss,
                pos_scale=pos_scale,
            )


# ------------------------------ Example ------------------------------ #
if __name__ == '__main__':
    kf = KalmanFilter(
        dt=1.0,
        proc_std_weight_pos=1 / 20,
        proc_std_weight_vel=1 / 100,
        meas_std_weight_pos=1 / 400,
        q_growth=1.05,
    )

    # Init with first detection.
    z0 = np.array([320.0, 200.0, 80.0, 160.0], dtype=np.float64)
    mean, cov = kf.initiate(z0)
    missed = 0

    # Frame 2: miss once. Advance a single gap of 1 (do NOT loop and also set missed_frames>0).
    mean, cov = kf.predict(mean, cov, missed_frames=1)
    missed += 1
    disp1 = kf.display_bbox(mean, cov, missed_count=missed)  # inflated
    print('After 1 miss:', disp1.tolist())

    # Frame 3: got a detection, gate then update.
    z1 = np.array([330.0, 205.0, 80.0, 160.0], dtype=np.float64)

    # Gate at 0.90 with 4 dof.
    z_pred, S, _ = kf.project(mean, cov)
    d = z1 - z_pred
    chol, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
    maha2 = np.sum(scipy.linalg.cho_solve((chol, lower), d, check_finite=False) * d)
    if maha2 <= CHI2_QUANTILES[0.90][4]:
        mean, cov = kf.update(mean, cov, z1)
        missed = 0  # reset misses after successful update

    disp2 = kf.display_bbox(mean, cov, missed_count=missed)  # no inflation, shrinks back
    print('After update:', disp2.tolist())

    # Jump 3 missed frames in one call (gap=3). Do not also loop per-frame.
    mean, cov = kf.predict(mean, cov, missed_frames=3)
    missed = 3
    disp3 = kf.display_bbox(mean, cov, missed_count=missed)
    print('After 3-miss jump:', disp3.tolist())
