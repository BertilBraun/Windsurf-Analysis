"""Discrete-optimization based multi-object tracker using ILP (Integer Linear Programming).

This implementation replaces Z3 with PuLP for solving the fragment linking optimization
problem. The core logic remains the same:

Local cost (adjacent frames only):
    For detections *i* (frame f) and *j* (frame f+1) on the **same track** we
    add *(1 – IoU(i,j))*.

Global embedding cohesion (sliding temporal window):
    For each detection *i* we look at up to K_prev previous and K_next future
    detections (by *time order*, not by track) whose frame difference is within
    a window of W frames (default W=10). When two detections *i,j* fall inside
    these limits **and are assigned to the same track**, we add an embedding
    distance cost `(1 - cos(i,j)) / N_i` where `N_i` is the number of neighbor
    comparisons originating from *i*. This normalises the expected contribution
    per detection so the overall magnitude is roughly independent of window
    size or clip length.

Other components preserved:
    • Binary decision variables for fragment links and track starts
    • Constraints ensuring each fragment has at most one incoming/outgoing link
    • Greedy pre-processing creates *must-link* groups (short obvious
      fragments) whose detections are forced to share a track id.

IMPORTANT ASSUMPTIONS
———————————————————
* Max per-frame detection count ≤ n_tracks else UNSAT.
* Every detection is assigned to some track (no unassigned sentinel).
* Embeddings are L2-normalised.
* Global embedding cost is quadratic in #detections (pairwise). If this becomes
  slow, restrict to a temporal window or subsample pairs.
"""

from __future__ import annotations
from dataclasses import dataclass

from typing import List, Optional

from ..visualization.stabilize import Transform

from ..util.video_io import VideoInfo
from ..common_types import Track

from ..settings import (
    MAX_OVERLAP_LENGTH_SECONDS,
    OPTIMIZER_W_START,
    OPTIMIZER_SHORT_MIN_LINK_IOU,
    OPTIMIZER_SHORT_MIN_LINK_COS,
    OPTIMIZER_SHORT_W_LINK_IOU,
    OPTIMIZER_SHORT_W_LINK_APP,
    OPTIMIZER_SHORT_W_LINK_GAP,
    OPTIMIZER_SHORT_LINK_COST_APPEARANCE_WINDOW_RADIUS,
    OPTIMIZER_LONG_MIN_LINK_IOU,
    OPTIMIZER_LONG_MIN_LINK_COS,
    OPTIMIZER_LONG_W_LINK_IOU,
    OPTIMIZER_LONG_W_LINK_APP,
    OPTIMIZER_LONG_W_LINK_GAP,
)

from .ILP_graph_solver import FragmentGraph, ILPGraphSolver


class DiscreteOptTracker:
    def __init__(
        self,
        # Track length thresholds
        short_track_threshold_seconds: float = 0.3,
        long_track_threshold_seconds: float = 1.0,
        # Short track parameters (< short_track_threshold_seconds)
        short_min_link_iou: float = OPTIMIZER_SHORT_MIN_LINK_IOU,
        short_min_link_cos: float = OPTIMIZER_SHORT_MIN_LINK_COS,
        short_w_link_iou: float = OPTIMIZER_SHORT_W_LINK_IOU,
        short_w_link_app: float = OPTIMIZER_SHORT_W_LINK_APP,
        short_w_link_gap: float = OPTIMIZER_SHORT_W_LINK_GAP,
        short_link_cost_appearance_window_radius: int = OPTIMIZER_SHORT_LINK_COST_APPEARANCE_WINDOW_RADIUS,
        # Long track parameters (> long_track_threshold_seconds)
        long_min_link_iou: float = OPTIMIZER_LONG_MIN_LINK_IOU,
        long_min_link_cos: float = OPTIMIZER_LONG_MIN_LINK_COS,
        long_w_link_iou: float = OPTIMIZER_LONG_W_LINK_IOU,
        long_w_link_app: float = OPTIMIZER_LONG_W_LINK_APP,
        long_w_link_gap: float = OPTIMIZER_LONG_W_LINK_GAP,
        long_link_cost_appearance_window_radius: int = 999999,  # Essentially infinite for mean appearance
        # Shared parameters
        w_start: float = OPTIMIZER_W_START,
    ):
        # Short track parameters
        self.short_min_link_iou = short_min_link_iou
        self.short_min_link_cos = short_min_link_cos
        self.short_w_link_iou = short_w_link_iou
        self.short_w_link_app = short_w_link_app
        self.short_w_link_gap = short_w_link_gap
        self.short_link_cost_appearance_window_radius = short_link_cost_appearance_window_radius

        # Long track parameters
        self.long_min_link_iou = long_min_link_iou
        self.long_min_link_cos = long_min_link_cos
        self.long_w_link_iou = long_w_link_iou
        self.long_w_link_app = long_w_link_app
        self.long_w_link_gap = long_w_link_gap
        self.long_link_cost_appearance_window_radius = long_link_cost_appearance_window_radius

        # Shared parameters
        self.w_start = w_start

        # Track length thresholds
        self.short_track_threshold_seconds = short_track_threshold_seconds
        self.long_track_threshold_seconds = long_track_threshold_seconds

    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        """Track objects using discrete optimization."""
        graph = self._build_fragment_graph(tracks, video_properties.fps)

        # Configure costs to match legacy behavior (free termination)
        N = len(graph.fragments)
        start_costs = [self.w_start] * N
        end_costs = [0.0] * N  # Legacy behavior: ending a track was free (implicit)
        max_cost_limit = self.w_start

        return ILPGraphSolver().optimize_graph(graph, start_costs, end_costs, max_cost_limit)[0]

    def _build_fragment_graph(self, fragments: List[Track], video_fps: int) -> FragmentGraph:
        """Build a graph of possible fragment connections with their costs."""
        fragments.sort(key=lambda x: x.start_frame)

        graph = FragmentGraph(fragments)
        N = len(fragments)

        for i, start_fragment in enumerate(fragments):
            for j in range(i, N):
                end_fragment = fragments[j]

                # Skip if fragments have overlapping frames
                gap = end_fragment.start_frame - start_fragment.end_frame
                if gap <= 0 or gap > video_fps * MAX_OVERLAP_LENGTH_SECONDS:
                    continue

                # Calculate connection cost
                cost = self._calculate_link_cost(start_fragment, end_fragment, video_fps)
                if cost is not None:
                    graph.add_connection(i, j, cost)

        return graph

    @dataclass
    class AdaptiveParameters:
        min_link_iou: float
        min_link_cos: float
        w_link_iou: float
        w_link_app: float
        w_link_gap: float
        window_radius: int

    def _get_adaptive_parameters(self, track_length_frames: int, video_fps: int) -> AdaptiveParameters:
        """Get adaptive parameters based on track length with linear interpolation."""
        short_track_threshold_frames = self.short_track_threshold_seconds * video_fps
        long_track_threshold_frames = self.long_track_threshold_seconds * video_fps

        # Determine interpolation factor
        if track_length_frames <= short_track_threshold_frames:
            # Short track - use short parameters
            alpha = 0.0
        elif track_length_frames >= long_track_threshold_frames:
            # Long track - use long parameters
            alpha = 1.0
        else:
            # Interpolate between short and long parameters
            alpha = (track_length_frames - short_track_threshold_frames) / (
                long_track_threshold_frames - short_track_threshold_frames
            )

        # Linear interpolation of parameters
        min_link_iou = self.short_min_link_iou + alpha * (self.long_min_link_iou - self.short_min_link_iou)
        min_link_cos = self.short_min_link_cos + alpha * (self.long_min_link_cos - self.short_min_link_cos)
        w_link_iou = self.short_w_link_iou + alpha * (self.long_w_link_iou - self.short_w_link_iou)
        w_link_app = self.short_w_link_app + alpha * (self.long_w_link_app - self.short_w_link_app)
        w_link_gap = self.short_w_link_gap + alpha * (self.long_w_link_gap - self.short_w_link_gap)

        # For window radius, interpolate but ensure it's an integer
        window_radius_float = self.short_link_cost_appearance_window_radius + alpha * (
            self.long_link_cost_appearance_window_radius - self.short_link_cost_appearance_window_radius
        )
        window_radius = int(window_radius_float)

        return self.AdaptiveParameters(min_link_iou, min_link_cos, w_link_iou, w_link_app, w_link_gap, window_radius)

    def _calculate_link_cost(self, start: Track, end: Track, video_fps: int) -> Optional[float]:
        """Calculate the cost of linking two tracks. Returns None if they can't be linked."""
        gap = end.start_frame - start.end_frame

        assert gap >= 0, 'Gap must be non-negative'
        assert gap <= video_fps * MAX_OVERLAP_LENGTH_SECONDS, 'Gap must be less than max overlap length'

        # Get adaptive parameters based on both tracks
        # Use the shorter DURATION (in frames) of both tracks for parameter selection,
        # not the number of detections. This better reflects temporal extent.
        start_duration_frames = start.end_frame - start.start_frame + 1
        end_duration_frames = end.end_frame - end.start_frame + 1
        min_duration_frames = min(start_duration_frames, end_duration_frames)

        adaptive_params = self._get_adaptive_parameters(min_duration_frames, video_fps)

        # Geometric similarity (IoU)
        iou = end.start.bbox.iou(start.end.bbox)

        if iou < adaptive_params.min_link_iou:
            return None

        # Appearance similarity
        if adaptive_params.window_radius >= min_duration_frames:  # Use mean appearance for long tracks
            cos = start.mean_embedding().distance(end.mean_embedding())
        else:
            cos = self._calculate_windowed_cosine_similarity(start, end, adaptive_params.window_radius)

        if cos < adaptive_params.min_link_cos:
            return None

        gap_percentage = gap / (video_fps * MAX_OVERLAP_LENGTH_SECONDS)

        # cost for similarity should be relative to the adaptive_params.min_link_cos i.e. if min_link_cos is 0.5, then cos_cost should be extremly high for cos close to 0.5 and only for cos close to 1.0 it should be 0
        cos_cost = 1.0 - (cos - adaptive_params.min_link_cos) / (1.0 - adaptive_params.min_link_cos)
        cos_cost = max(0.0, cos_cost)

        # Calculate total cost
        cost = (
            adaptive_params.w_link_iou * (1.0 - iou)
            + adaptive_params.w_link_app * cos_cost
            + adaptive_params.w_link_gap * gap_percentage
        )

        return cost

    def _calculate_windowed_cosine_similarity(self, start: Track, end: Track, window_radius: int) -> float:
        """Calculate average cosine similarity over a temporal window."""
        n_pairs = 0
        cos_sum = 0.0

        for i in range(-window_radius, window_radius + 1):
            d1 = start.detections_by_frame.get(start.end_frame + i)
            if d1 is None:
                continue

            for j in range(-window_radius, window_radius + 1):
                d2 = end.detections_by_frame.get(end.start_frame + j)
                if d2 is None:
                    continue

                cos = d1.embedding.distance(d2.embedding)
                cos_sum += cos
                n_pairs += 1

        return cos_sum / n_pairs if n_pairs > 0 else 0.0
