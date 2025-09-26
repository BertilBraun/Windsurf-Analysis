from __future__ import annotations


from typing import List, Tuple

import math
import numpy as np

from server.inference.src.visualization.stabilize import Transform

from ..util.video_io import VideoInfo
from ..common_types import BoundingBox, Detection, Track

from ..settings import MAX_OVERLAP_LENGTH_SECONDS, OPTIMIZER_W_START

from .ILP_graph_solver import FragmentGraph, ILPGraphSolver
from server.inference.bot_sort.kalman_filter import KalmanFilter

EPS = 1e-9


class IterativeILPTracker:
    """Iterative ILP tracker.

    Plan:
    1. Build a graph of possible fragment connections with their costs.
        - Possible connections (A -> B) are based on:
            - A.start_frame < B.start_frame
            - intersection(A.frames, B.frames) == empty
            - gap between A.end_frame and B.start_frame <= MAX_OVERLAP_LENGTH_SECONDS * video_fps
        - Costs are based on:
            To calculate the actual cost, we use the sum of the NLL for motion, appearance and gap.
            - Motion: KF tracking + GMC: we start a KF filter at A.start then we continue frame by frame, either, if A has the current frame, update the KF filter, or if B has the current frame, add the kf.gating_distance to the total cost. Once we reach max(A.end_frame, B.end_frame), we add divide the total cost by |B.frames| to get the average gating distance. We apply the appropriate Camera Transforms on each frame. Transfors are defined as: Transform = NamedTuple('Transform', [('dx', float), ('dy', float), ('da', float), ('frame_idx', int)]) # dx, dy, da for each frame relative to the previous frame.
            - Appearance: embedding is a LAB color histogram, we compute the mean histogram for both A and B and then use the chi-squared distance to get the appearance similarity probability by calculating platt_prob_from_dist.
            - Gap: tbd.. Something with a p_miss probability.
    2. Solve the ILP problem with a pretty low start_cost (no need to link up everything - it's fine to have some split tracks or even unassigned detections - it's iterative).
    3. Repeat from Step 1. but this time with the solution of the previous iteration as the starting point. We increase the start_cost by a small amount each time.
    4. Stop when the solution is stable (i.e. the cost of the solution is not changing much) or we have reached a maximum number of iterations (4 iterations).
    5. Return the solution.
    """

    def __init__(
        self,
        w_start: float = OPTIMIZER_W_START,
        motion_weight: float = 1.0,
        appearance_weight: float = 1.0,
        gap_weight: float = 0.05,
        motion_use_only_position: bool = True,
        gap_p_miss: float = 0.98,
        appearance_a: float = 7.427,
        appearance_b: float = 4.088,
    ) -> None:
        self.w_start = float(w_start)
        self.motion_weight = float(motion_weight)
        self.appearance_weight = float(appearance_weight)
        self.gap_weight = float(gap_weight)
        self.motion_use_only_position = bool(motion_use_only_position)
        self.gap_p_miss = float(gap_p_miss)
        self.appearance_a = float(appearance_a)
        self.appearance_b = float(appearance_b)

    # ───────────────────────────────── public API ───────────────────────────────── #

    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        """Track objects using discrete optimization."""
        graph = self._build_fragment_graph(tracks, video_properties.fps, transforms)
        return ILPGraphSolver(self.w_start).optimize_graph(graph)

    # ─────────────────────────────── graph building ────────────────────────────── #

    def _build_fragment_graph(
        self, fragments: list[Track], video_fps: int, transforms: list[Transform]
    ) -> FragmentGraph:
        """Build a graph of possible fragment connections with their costs."""
        fragments.sort(key=lambda x: x.start_frame)

        graph = FragmentGraph(fragments)
        N = len(fragments)

        # Pre-compute cumulative camera motion (translation only) for stabilised coordinates
        cum_dx, cum_dy = self._compute_cumulative_camera_offsets(transforms)

        for i, start_fragment in enumerate(fragments):
            for j in range(i, N):
                end_fragment = fragments[j]

                # Skip if fragments have overlapping frames or are out of max gap
                gap = end_fragment.start_frame - start_fragment.end_frame
                if gap <= 0 or gap > video_fps * MAX_OVERLAP_LENGTH_SECONDS:
                    continue

                # Calculate connection cost
                cost = self._calculate_link_cost(start_fragment, end_fragment, video_fps, cum_dx, cum_dy)
                if cost is not None:
                    graph.add_connection(i, j, cost)

        return graph

    # ──────────────────────────────── cost helpers ─────────────────────────────── #

    def _calculate_link_cost(
        self,
        start: Track,
        end: Track,
        video_fps: int,
        cum_dx: List[float],
        cum_dy: List[float],
    ) -> float | None:
        """Calculate total NLL cost for linking two track fragments.

        Returns None if the pair should be disallowed (e.g., numerical issues).
        """
        gap = end.start_frame - start.end_frame
        if gap < 0 or gap > video_fps * MAX_OVERLAP_LENGTH_SECONDS:
            return None

        # Motion model NLL using Kalman filter with stabilised coordinates
        motion_nll = self._motion_nll(start, end, cum_dx, cum_dy)
        if math.isnan(motion_nll) or math.isinf(motion_nll):
            return None

        # Appearance NLL using chi-squared distance + Platt scaling
        appearance_nll = self._appearance_nll(start, end)
        if math.isnan(appearance_nll) or math.isinf(appearance_nll):
            return None

        # Gap penalty as NLL with per-frame miss probability
        gap_nll = self._gap_nll(gap)

        total_cost = (
            self.motion_weight * motion_nll + self.appearance_weight * appearance_nll + self.gap_weight * gap_nll
        )
        return float(total_cost)

    # ───────────────────────────────── appearance ──────────────────────────────── #

    def _appearance_nll(self, a: Track, b: Track) -> float:
        chi2 = chi2_dist(a.mean_embedding(), b.mean_embedding())
        p = platt_prob_from_dist(chi2, self.appearance_a, self.appearance_b)
        return NLL(p)

    # ─────────────────────────────────── motion ────────────────────────────────── #

    def _motion_nll(
        self, a: Track, b: Track, cum_dx: List[float], cum_dy: List[float]
    ) -> float:  # average 0.5 * maha over B frames
        if not b.sorted_detections:
            return 0.0

        # Build per-frame measurements in stabilised coordinates
        def to_xywh_stab(det: Detection) -> np.ndarray:
            x, y, w, h = self._bbox_to_xywh(det.bbox)
            fx = det.frame_idx
            assert fx < len(cum_dx) and fx < len(cum_dy), f'Frame {fx} is out of bounds for cum_dx and cum_dy'
            offx, offy = cum_dx[fx], cum_dy[fx]
            return np.array([x - offx, y - offy, w, h], dtype=np.float64)

        kf = KalmanFilter()

        # Initialise with A.start in stabilised frame
        mean, cov = kf.initiate(to_xywh_stab(a.start))

        f = a.start_frame
        f_end = max(a.end_frame, b.end_frame)

        total_maha = 0.0
        b_count = 0

        while f <= f_end:
            if f > a.start_frame:
                mean, cov = kf.predict(mean, cov, missed_frames=1)

            # If A has a detection at this frame, incorporate it
            det_a = a.detections_by_frame.get(f)
            if det_a is not None:
                meas_a = to_xywh_stab(det_a)
                mean, cov = kf.update(mean, cov, meas_a)

            # If B has a detection at this frame, accumulate gating distance
            det_b = b.detections_by_frame.get(f)
            if det_b is not None:
                meas_b = to_xywh_stab(det_b)
                d = kf.gating_distance(
                    mean,
                    cov,
                    measurements=meas_b[None, :],
                    only_position=self.motion_use_only_position,
                    metric='maha',
                )
                total_maha += float(d[0])
                b_count += 1

            f += 1

        if b_count == 0:
            return 1e6  # Shouldn't happen, but keep this pair unattractive

        avg_maha = total_maha / b_count
        # For Gaussian likelihood, NLL ≈ 0.5 * maha (+const). Drop constant.
        return 0.5 * avg_maha

    def _bbox_to_xywh(self, bbox: BoundingBox) -> Tuple[float, float, float, float]:
        x = (bbox.x1 + bbox.x2) / 2.0
        y = (bbox.y1 + bbox.y2) / 2.0
        w = max(EPS, float(bbox.x2 - bbox.x1))
        h = max(EPS, float(bbox.y2 - bbox.y1))
        return x, y, w, h

    # ───────────────────────────────────── gap ─────────────────────────────────── #

    def _gap_nll(self, gap_frames: int) -> float:
        # Independent miss probability per frame; NLL = -log(p_miss^gap) = gap * -log(p_miss)
        gap_frames = int(max(0, gap_frames))
        if gap_frames == 0:
            return 0.0
        return NLL(self.gap_p_miss**gap_frames)

    # ───────────────────────────────── transforms ──────────────────────────────── #

    def _compute_cumulative_camera_offsets(self, transforms: List[Transform]) -> Tuple[List[float], List[float]]:
        if not transforms:
            return [], []

        transforms_sorted = sorted(transforms, key=lambda t: int(t.frame_idx))
        max_frame = int(transforms_sorted[-1].frame_idx)
        cum_dx = [0.0] * (max_frame + 1)
        cum_dy = [0.0] * (max_frame + 1)

        last_f = int(transforms_sorted[0].frame_idx)
        # Ensure we propagate initial index correctly
        for f in range(0, last_f + 1):
            cum_dx[f] = float(transforms_sorted[0].dx)
            cum_dy[f] = float(transforms_sorted[0].dy)

        acc_x = 0.0
        acc_y = 0.0
        for t in transforms_sorted:
            f = int(t.frame_idx)
            acc_x += float(t.dx)
            acc_y += float(t.dy)
            if f < len(cum_dx):
                cum_dx[f] = acc_x
                cum_dy[f] = acc_y

        # Fill gaps by carrying forward last known value
        for f in range(1, len(cum_dx)):
            if cum_dx[f] == 0.0 and cum_dy[f] == 0.0 and f not in {int(t.frame_idx) for t in transforms_sorted}:
                cum_dx[f] = cum_dx[f - 1]
                cum_dy[f] = cum_dy[f - 1]

        return cum_dx, cum_dy


def chi2_dist(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """Calculate the chi2 distance between two embeddings."""
    num = (p - q) ** 2
    den = p + q + eps
    return 0.5 * float((num / den).sum())


def platt_prob_from_dist(d: float, a: float, b: float) -> float:
    """Calculate the probability for a distance to say, that the two tracks are the same. `a` and `b` are parameters of the platt scaling. The returned probability is in the range [0, 1] (sigmoid(a * -d + b))"""
    return sigmoid(a * (-d) + b)


def sigmoid(z: float) -> float:
    """Calculate the sigmoid of a value."""
    return 1.0 / (1.0 + np.exp(-z))


def clamp_prob(p: float) -> float:
    """Clamp a probability to the range [EPS, 1 - EPS]."""
    return max(EPS, min(1 - EPS, p))


def NLL(p: float) -> float:
    """Calculate the negative log-likelihood of a probability."""
    p = clamp_prob(p)
    return float(-math.log(p / (1 - p)))  # TODO check if this is correct
