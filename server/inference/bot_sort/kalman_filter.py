# kalman_tracking.py
# Typed, commented Kalman filter for bbox tracking with strong-trust detections,
# configurable Mahalanobis gating, and uncertainty-inflated “search” boxes.

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt
import scipy.linalg


NDArrayF = npt.NDArray[np.float64]


# Chi-square quantile table used for Mahalanobis gating and uncertainty inflation.
# Keys are confidence alpha, values are per degrees-of-freedom thresholds.
CHI2_QUANTILES: Dict[float, Dict[int, float]] = {
    0.90: {1: 2.7055, 2: 4.6052, 3: 6.2514, 4: 7.779},
    0.95: {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877},
}


class KalmanFilter:
    """
    8D state: [x, y, w, h, vx, vy, vw, vh].
    Motion: constant velocity.
    Measurement: direct observation of [x, y, w, h].

    Design goals for “trust detections”:
      - Small measurement noise R via _meas_std_weight_pos (fraction of box size).
      - Reasonable process noise Q via _proc_*.
      - Optional growth factor _q_growth for long gaps.
      - Gating with Mahalanobis distance and configurable chi-square quantile.
      - Uncertainty-inflated output bbox for search/visualization.
    """

    def __init__(
        self,
        dt: float = 1.0,
        proc_std_weight_pos: float = 1.0 / 20.0,
        proc_std_weight_vel: float = 1.0 / 100.0,
        meas_std_weight_pos: float = 1.0 / 400.0,
        q_growth: float = 1.2,
    ) -> None:
        """
        Args:
            dt: timestep in frames.
            proc_std_weight_pos: Q position std as fraction of [w, h].
            proc_std_weight_vel: Q velocity std as fraction of [w, h].
            meas_std_weight_pos: R std as fraction of [w, h]. Make small to trust detections.
            q_growth: multiplicative growth per missed frame applied to Q.
        """
        ndim = 4
        self.ndim = ndim
        self.dt = float(dt)

        # Constant-velocity state transition.
        self._motion_mat: NDArrayF = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = self.dt

        # Measurement matrix maps state -> measurement.
        self._update_mat: NDArrayF = np.eye(ndim, 2 * ndim, dtype=np.float64)

        # Noise hyperparameters.
        self._proc_std_weight_pos = float(proc_std_weight_pos)
        self._proc_std_weight_vel = float(proc_std_weight_vel)
        self._meas_std_weight_pos = float(meas_std_weight_pos)
        self._q_growth = float(q_growth)

    # ---------------------------- Core API ---------------------------- #

    def initiate(self, measurement: NDArrayF) -> Tuple[NDArrayF, NDArrayF]:
        """
        Initialize from first detection z = [x, y, w, h].

        Returns:
            mean (8,), covariance (8,8).
        """
        mean_pos = measurement.astype(np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Larger initial uncertainty than steady-state Q/R.
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
        # Process noise Q as function of current box size and missed frames.
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
        q_scale = self._q_growth ** max(missed_frames, 0)
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])) * q_scale

        mean = mean @ self._motion_mat.T
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: NDArrayF, covariance: NDArrayF) -> Tuple[NDArrayF, NDArrayF]:
        """
        Project state to measurement space with measurement noise R.

        Returns:
            z_mean (4,), S = HPHT + R (4,4)
        """
        # Measurement noise R: small to trust detections.
        std = np.array(
            [
                self._meas_std_weight_pos * max(mean[2], 1e-6),  # x std scales with w
                self._meas_std_weight_pos * max(mean[3], 1e-6),  # y std scales with h
                self._meas_std_weight_pos * max(mean[2], 1e-6),  # w
                self._meas_std_weight_pos * max(mean[3], 1e-6),  # h
            ],
            dtype=np.float64,
        )
        innovation_cov = np.diag(np.square(std))

        z_mean = self._update_mat @ mean
        S = self._update_mat @ covariance @ self._update_mat.T
        return z_mean, S + innovation_cov

    def update(self, mean: NDArrayF, covariance: NDArrayF, measurement: NDArrayF) -> Tuple[NDArrayF, NDArrayF]:
        """
        Measurement update with z = [x, y, w, h].

        Returns:
            updated mean, covariance
        """
        z_pred, S = self.project(mean, covariance)
        # Stable gain via Cholesky of S.
        chol, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve((chol, lower), (covariance @ self._update_mat.T).T, check_finite=False).T
        innovation = measurement - z_pred

        new_mean = mean + K @ innovation
        new_covariance = covariance - K @ S @ K.T
        return new_mean, new_covariance

    # ------------------------- Association utils ------------------------- #

    @staticmethod
    def chi2_threshold(dof: int, alpha: float = 0.95) -> float:
        """Return chi-square threshold for given dof and confidence."""
        if alpha not in CHI2_QUANTILES or dof not in CHI2_QUANTILES[alpha]:
            raise ValueError(f'No chi-square entry for alpha={alpha}, dof={dof}')
        return CHI2_QUANTILES[alpha][dof]

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
        z_pred, S = self.project(mean, covariance)
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
        mean: NDArrayF,  # shape (N, 8)
        covariance: NDArrayF,  # shape (N, 8, 8)
        missed_frames: Optional[int | npt.NDArray[np.int_]] = 0,
    ) -> Tuple[NDArrayF, NDArrayF]:
        """
        Vectorized one-step prediction for N tracks.

        Args:
            mean: (N,8)
            covariance: (N,8,8)
            missed_frames: scalar int or (N,) array of ints. Scales Q per track.
                           To advance X frames without updates, call this method
                           X times in a loop (so state advances and uncertainty compounds).

        Returns:
            mean_pred (N,8), cov_pred (N,8,8)
        """
        N = mean.shape[0]
        assert covariance.shape == (N, 8, 8)

        w = np.maximum(mean[:, 2], 1e-6)
        h = np.maximum(mean[:, 3], 1e-6)

        std_pos = np.stack(
            [
                self._proc_std_weight_pos * w,
                self._proc_std_weight_pos * h,
                self._proc_std_weight_pos * w,
                self._proc_std_weight_pos * h,
            ],
            axis=1,
        )  # (N,4)
        std_vel = np.stack(
            [
                self._proc_std_weight_vel * w,
                self._proc_std_weight_vel * h,
                self._proc_std_weight_vel * w,
                self._proc_std_weight_vel * h,
            ],
            axis=1,
        )  # (N,4)

        std_all = np.concatenate([std_pos, std_vel], axis=1)  # (N,8)

        if isinstance(missed_frames, np.ndarray):
            q_scale = np.power(self._q_growth, np.maximum(missed_frames.astype(np.int64), 0))
        else:
            q_scale = np.full((N,), self._q_growth ** int(missed_frames or 0), dtype=np.float64)

        var_all = (std_all**2) * q_scale[:, None]  # (N,8)

        # Build batched diagonal motion covariances.
        motion_cov = np.zeros((N, 8, 8), dtype=np.float64)
        idx = np.arange(8)
        motion_cov[:, idx, idx] = var_all

        # Predict means.
        mean_pred = mean @ self._motion_mat.T  # (N,8)

        # Predict covariances: P' = F P F^T + Q  (batched)
        left = np.einsum('ij,njk->nik', self._motion_mat, covariance)  # (N,8,8)
        cov_pred = left @ self._motion_mat.T + motion_cov  # (N,8,8)

        return mean_pred, cov_pred

    # ---------------------- Uncertainty-inflated box ---------------------- #

    def inflated_bbox(
        self,
        mean: NDArrayF,
        covariance: NDArrayF,
        alpha: float = 0.95,
        include_size_unc: bool = True,
    ) -> NDArrayF:
        """
        Axis-aligned bbox inflated by positional (and optional size) uncertainty.

        Args:
            mean, covariance: state and covariance after predict().
            alpha: chi-square quantile for coverage of uncertainty.
            include_size_unc: if True, add terms from S[2,2], S[3,3].

        Returns:
            [x, y, w_out, h_out] covering uncertainty at confidence alpha.
        """
        z_mean, S = self.project(mean, covariance)

        # Positional ellipse -> axis-aligned bounds.
        k_pos = np.sqrt(self.chi2_threshold(dof=2, alpha=alpha))
        rx = np.sqrt(max(S[0, 0], 0.0)) * k_pos
        ry = np.sqrt(max(S[1, 1], 0.0)) * k_pos

        x, y, w, h = z_mean
        w_out = float(w + 2.0 * rx)
        h_out = float(h + 2.0 * ry)

        if include_size_unc:
            k_size = np.sqrt(self.chi2_threshold(dof=1, alpha=alpha))
            w_out += k_size * np.sqrt(max(S[2, 2], 0.0))
            h_out += k_size * np.sqrt(max(S[3, 3], 0.0))

        return np.array([float(x), float(y), max(w_out, 1e-3), max(h_out, 1e-3)], dtype=np.float64)


# ------------------------------ Example use ------------------------------ #
# The block below is illustrative. Remove or adapt in production.

if __name__ == '__main__':
    kf = KalmanFilter(
        dt=1.0,
        proc_std_weight_pos=1 / 20,
        proc_std_weight_vel=1 / 100,
        meas_std_weight_pos=1 / 400,  # small => trust detections
        q_growth=1.2,
    )

    # First detection
    z0 = np.array([320.0, 200.0, 80.0, 160.0], dtype=np.float64)
    mean, cov = kf.initiate(z0)

    # Predict without update (missed detection)
    mean, cov = kf.predict(mean, cov, missed_frames=1)
    inflated = kf.inflated_bbox(mean, cov, alpha=0.95, include_size_unc=True)
    print('Inflated search bbox after 1 miss:', inflated.tolist())

    # Next frame: a candidate detection appears
    z1 = np.array([330.0, 205.0, 81.0, 160.0], dtype=np.float64)

    # Gate with Mahalanobis at alpha=0.90, dof=4
    d2 = kf.gating_distance(mean, cov, measurements=z1[None, :], only_position=False, metric='maha')[0]
    gate = KalmanFilter.chi2_threshold(dof=4, alpha=0.90)
    if d2 <= gate:
        mean, cov = kf.update(mean, cov, z1)
        print('Update accepted. New mean:', mean[:4].tolist())
    else:
        print('Detection rejected by gate.')
