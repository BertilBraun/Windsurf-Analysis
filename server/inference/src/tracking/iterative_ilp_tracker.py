from __future__ import annotations


from typing import Dict, Tuple

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
        max_consecutive_b_frames: int = 5,
    ) -> None:
        self.w_start = float(w_start)
        self.motion_weight = float(motion_weight)
        self.appearance_weight = float(appearance_weight)
        self.gap_weight = float(gap_weight)
        self.motion_use_only_position = bool(motion_use_only_position)
        self.gap_p_miss = float(gap_p_miss)
        self.appearance_a = float(appearance_a)
        self.appearance_b = float(appearance_b)
        self.max_consecutive_b_frames = int(max(0, max_consecutive_b_frames))

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

        # Build per-frame delta warp map consistent with BoTSORT (prev -> curr)
        frame_warp: Dict[int, Tuple[float, float, float, float]] = {}
        for t in transforms:
            f = int(t.frame_idx)
            c = math.cos(float(t.da))
            s = math.sin(float(t.da))
            frame_warp[f] = (c, s, float(t.dx), float(t.dy))

        for i, start_fragment in enumerate(fragments):
            for j in range(i, N):
                end_fragment = fragments[j]

                # Skip if fragments have overlapping frames or are out of max gap
                gap = end_fragment.start_frame - start_fragment.end_frame
                start_frames = set(start_fragment.detections_by_frame.keys())
                end_frames = set(end_fragment.detections_by_frame.keys())
                frames_overlap = start_frames & end_frames
                if gap > video_fps * MAX_OVERLAP_LENGTH_SECONDS or len(frames_overlap) > 0:
                    continue

                # Calculate connection cost
                cost = self._calculate_link_cost(start_fragment, end_fragment, frame_warp)
                if cost is not None:
                    graph.add_connection(i, j, cost)

        return graph

    # ──────────────────────────────── cost helpers ─────────────────────────────── #

    def _calculate_link_cost(
        self,
        start: Track,
        end: Track,
        frame_warp: Dict[int, Tuple[float, float, float, float]],
    ) -> float | None:
        """Calculate total NLL cost for linking two track fragments.

        Returns None if the pair should be disallowed (e.g., numerical issues).
        """
        # Motion model NLL using Kalman filter with stabilised coordinates
        motion_nll = self._motion_nll(start, end, frame_warp)
        if math.isnan(motion_nll) or math.isinf(motion_nll):
            return None

        # Appearance NLL using chi-squared distance + Platt scaling
        appearance_nll = self._appearance_nll(start, end)
        if math.isnan(appearance_nll) or math.isinf(appearance_nll):
            return None

        # Gap penalty as NLL with per-frame miss probability
        gap = (
            max(start.end_frame, end.end_frame)
            - start.start_frame
            - len(start.sorted_detections)
            - len(end.sorted_detections)
        )
        gap_nll = self._gap_nll(gap)

        return self.motion_weight * motion_nll + self.appearance_weight * appearance_nll + self.gap_weight * gap_nll

    # ───────────────────────────────── appearance ──────────────────────────────── #

    def _appearance_nll(self, a: Track, b: Track) -> float:
        chi2 = chi2_dist(a.mean_embedding(), b.mean_embedding())
        p = platt_prob_from_dist(chi2, self.appearance_a, self.appearance_b)
        return NLL(p)

    # ─────────────────────────────────── motion ────────────────────────────────── #

    def _motion_nll(
        self,
        a: Track,
        b: Track,
        frame_warp: Dict[int, Tuple[float, float, float, float]],
    ) -> float:
        if not b.sorted_detections:
            return 0.0

        # Measurements in image coordinates
        def to_xywh(det: Detection) -> np.ndarray:
            x, y, w, h = self._bbox_to_xywh(det.bbox)
            return np.array([x, y, w, h], dtype=np.float64)

        kf = KalmanFilter()

        # Initialise with A.start in image coordinates
        mean, cov = kf.initiate(to_xywh(a.start))

        total_maha = 0.0
        b_count = 0

        frame_indices = list(sorted(list(a.detections_by_frame.keys()) + list(b.detections_by_frame.keys())))
        # remove elements K after A.end_frame
        frame_indices = frame_indices[: frame_indices.index(a.end_frame) + 1 + self.max_consecutive_b_frames]

        for i, f in enumerate(frame_indices):
            if f > a.start_frame:
                mean, cov = kf.predict(mean, cov, missed_frames=1)
                # Apply forward per-frame warp (prev -> curr), consistent with BoTSORT
                assert f in frame_warp, f'Frame {f} not in frame_warp'
                c, s, dx, dy = frame_warp[int(f)]
                A = np.eye(8, dtype=np.float64)
                A[0:2, 0:2] = np.array([[c, -s], [s, c]], dtype=np.float64)
                A[4:6, 4:6] = np.array([[c, -s], [s, c]], dtype=np.float64)
                mean = A @ mean
                mean[0] += dx
                mean[1] += dy
                cov = A @ cov @ A.T

            # If A has a detection at this frame, incorporate it
            det_a = a.detections_by_frame.get(f)
            if det_a is not None:
                mean, cov = kf.update(mean, cov, to_xywh(det_a))

            # If B has a detection at this frame, accumulate gating distance
            det_b = b.detections_by_frame.get(f)
            if det_b is not None:
                # if a has any detection in the past K frames, we can continue, else skip
                if not any(a.detections_by_frame.get(f) for f in frame_indices[i - self.max_consecutive_b_frames : i]):
                    continue
                meas_b = to_xywh(det_b)
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
        return 0.5 * avg_maha  # TODO to probability

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
