from __future__ import annotations

from typing import Dict, Literal, Tuple, List, Optional

import math
import numpy as np

from server.inference.src.motion.cmc import CMC
from server.inference.src.util.algebra import probability_from_dist
from server.inference.src.visualization.debug.session import DebugSession


from ..util.video_io import VideoInfo
from ..common_types import Detection, Track, TrackId
from ..settings import MAX_OVERLAP_LENGTH_SECONDS, EPS
from .ILP_graph_solver import FragmentGraph, ILPGraphSolver
from server.inference.src.motion.kalman_filter import KFState
from server.inference.src.visualization.stabilize import Transform
from server.inference.src.visualization.debug import get_debug_session
from server.inference.src.visualization.debug.graph import EdgeRecord, show_graph_interactive
from server.inference.src.visualization.debug.overlays import EdgeMetrics, compose_fragment_pair_view


class IterativeILPTracker:
    """Iterative ILP tracker.

    Plan:
    1. Build a graph of possible fragment connections with their costs.
        - Possible connections (A -> B) are based on:
            - A.end_frame < B.start_frame and B.start_frame <= A.end_frame + MAX_OVERLAP_LENGTH_SECONDS * video_fps
        - Costs are based on:
            To calculate the actual cost, we use the sum of the NLL for motion, appearance and gap.
            - Motion: KF Mahalanobis NLL + GMC (position-only) + 0.5*log|S_pos|. We apply the appropriate Camera Transforms on each frame. Transfors are defined as: Transform = NamedTuple('Transform', [('dx', float), ('dy', float), ('da', float), ('frame_idx', int)]) # dx, dy, da for each frame relative to the previous frame.
            - Appearance: embedding is a LAB color histogram, we compute the mean histogram for both A and B and then use the chi-squared distance to get the appearance similarity probability by calculating platt_prob_from_dist. Lab χ² → Platt prob → NLLR
            - Gap: per-frame miss NLL
    2. Solve the ILP problem with a pretty low start_cost (no need to link up everything - it's fine to have some split tracks or even unassigned detections - it's iterative).
    3. Repeat from Step 1. but this time with the solution of the previous iteration as the starting point. We increase the start_cost by a small amount each time.
    4. Stop when the solution is stable (i.e. the cost of the solution is not changing much) or we have reached a maximum number of iterations (4 iterations).
    5. Return the solution.
    """

    def __init__(
        self,
        video_path: str | None = None,
        # Start-cost schedule (controls edge cap and start penalty in ILP)
        w_start: float = 83.18548233891453,  # initial start cost (backward-compatible)
        start_cost_mode: Literal['linear', 'geo'] = 'linear',  # 'linear' or 'geo'
        start_cost_growth: float = 7.075710523532933,  # linear: additive step; geo: multiplicative factor
        start_cost_max: Optional[float] = 76.21143004790967,  # optional cap
        # Per-term cost weights
        w_motion: float = 2.4792939452722207,
        w_appearance: float = 0.43588112965420484,
        w_gap: float = 5.689009868203195,
        # Gap model
        p_miss: float = 0.8859392091825458,  # for gap NLL
        # Motion evaluation
        max_detections_to_compare: int = 2,  # eval first K detections of B (1..3 recommended)
        use_position_only: bool = False,  # gating_distance on (cx,cy) or (cx,cy,w,h)
        # Appearance similarity
        appearance_similarity_gamma: float = 10,
        # Iteration & stopping
        max_optimization_iterations: int = 5,
        # optional splitting rules
        enable_splitting: bool = True,
        internal_split_gap_frames: int = 2,  # 0 disables; >0 splits tracks on internal gaps > this
        motion_split_nll_max: float = 0.31442373897384446,  # split if P_same(motion) < this
        appearance_split_nll_max: float = 2.1193150519036594,  # split if P_same(app) < this
        appearance_split_window: int = 10,  # mean of last W embeddings vs current
        max_splits_per_track: Optional[int] = 8,
    ) -> None:
        self.video_path = video_path
        # Start-cost scheduling
        assert start_cost_mode in ['linear', 'geo']
        assert start_cost_growth > 0.0
        self.w_start_initial = w_start
        self.start_cost_mode = start_cost_mode
        self.start_cost_growth = start_cost_growth
        self.start_cost_max = start_cost_max
        # Term weights
        self.w_motion = float(w_motion)
        self.w_appearance = float(w_appearance)
        self.w_gap = float(w_gap)
        # Gap model
        self.p_miss = float(p_miss)
        # Motion evaluation
        self.max_detections_to_compare = int(max(1, max_detections_to_compare))
        self.use_position_only = bool(use_position_only)
        # Appearance similarity
        self.appearance_similarity_gamma = appearance_similarity_gamma
        # Iteration & stopping
        self.max_optimization_iterations = int(max(1, max_optimization_iterations))
        # Splitting
        self.enable_splitting = bool(enable_splitting)
        self.internal_split_gap_frames = int(max(0, internal_split_gap_frames))
        self.motion_split_nll_max = motion_split_nll_max
        self.appearance_split_nll_max = appearance_split_nll_max
        self.appearance_split_window = int(max(1, appearance_split_window))
        self.max_splits_per_track = None if max_splits_per_track is None else int(max(0, max_splits_per_track))

        # TODO remove
        # Debugging/visualization
        self.enable_edge_logging = False
        self.enable_graph_viz = False
        self.inline_edge_windows = False

        # Storage for debug visualization
        self._edge_debug_records: List[EdgeRecord] = []

    # ───────────────────────────────── public API ───────────────────────────────── #
    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        """Run iterative graph building and ILP solve. Stops when cost does not improve."""
        return self._internal_track_with_iteration_returned(tracks, video_properties, transforms)[0]

    def _internal_track_with_iteration_returned(
        self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]
    ) -> tuple[List[Track], float]:
        """Run iterative graph building and ILP solve. Stops when cost does not improve."""

        cmc = CMC(transforms)
        max_frame_gap = int(round(MAX_OVERLAP_LENGTH_SECONDS * video_properties.fps))

        # Initialize debug session if any debug feature is enabled
        debug_enabled = bool(self.enable_edge_logging or self.enable_graph_viz)
        debug = get_debug_session(self.video_path or '', enabled=debug_enabled)

        tracks = self._maybe_split_tracks(tracks, cmc)

        best_cost = float('inf')
        iteration = 0

        for iteration in range(self.max_optimization_iterations):
            # Iteration-level debug: list fragments/ids before building the graph
            self._edge_debug_records = []

            graph = self._build_fragment_graph(tracks, max_frame_gap, cmc)

            self._debug_display(debug, graph, iteration, cmc)
            self._log_tracks(iteration, tracks, 'fragments before graph build')

            # Solve with scheduled start cost
            scheduled_w_start = self._scheduled_start_cost(iteration)
            new_tracks, new_cost = ILPGraphSolver(scheduled_w_start).optimize_graph(graph)

            self._log_tracks(iteration, new_tracks, 'new tracks after ILP solve')

            if new_cost >= best_cost:  # no improvement → stop
                self._log(f'No improvement in iteration {iteration}, stopping. {new_cost} >= {best_cost}')
                break

            # Assign new tracks and update best cost
            last_tracks, tracks, best_cost = tracks, new_tracks, new_cost

            are_all_tracks_merged = _are_all_tracks_merged(new_tracks, max_frame_gap)
            no_assignment_changes = _are_assignments_the_same(last_tracks, new_tracks)
            if are_all_tracks_merged or no_assignment_changes:
                self._log(
                    f'All tracks merged or no assignment changes, stopping. Cost: {new_cost} Are all tracks merged: {are_all_tracks_merged} No assignment changes: {no_assignment_changes}'
                )
                break

            # optional: after each solve, split again if enabled
            tracks = self._maybe_split_tracks(tracks, cmc)

        debug.close()

        # Remerge tracks with same track id in case of internal splits
        return _remerge_tracks(tracks), iteration

    # ─────────────────────────────── graph building ────────────────────────────── #

    def _build_fragment_graph(self, fragments: List[Track], max_frame_gap: int, cmc: CMC) -> FragmentGraph:
        """Build forward edges with costs. Logs one line per edge if enabled.
        If `enable_edge_logging` is True, also displays a debug visualization for each edge candidate
        right before adding the connection to the graph.
        """
        fragments = sorted(fragments, key=lambda t: t.start_frame)
        graph = FragmentGraph(fragments)

        # KF cache: fit once per fragment (end state after all its detections)
        kf_cache: Dict[TrackId, KFState] = {
            i: KFState.fit_kf_end_state(frag.sorted_detections, cmc) for i, frag in enumerate(fragments)
        }

        for i, j in _possible_mergeable_candidates(fragments, max_frame_gap):
            A = fragments[i]
            B = fragments[j]
            # compute costs
            motion_nll = _motion_nll(kf_cache[i], B, cmc, self.max_detections_to_compare, self.use_position_only)
            if math.isinf(motion_nll) or math.isnan(motion_nll):
                continue

            appearance_nll = _appearance_nll(A, B, self.appearance_similarity_gamma)
            if math.isinf(appearance_nll) or math.isnan(appearance_nll):
                continue

            gap_frames = A.frame_gap(B)
            gap_nll = _gap_nll(gap_frames, self.p_miss)

            # Weighted sum of costs
            total = self.w_motion * motion_nll + self.w_appearance * appearance_nll + self.w_gap * gap_nll

            graph.add_connection(i, j, total)

            # Collect per-edge debug data for interactive visualization
            if self.enable_graph_viz:
                self._edge_debug_records.append(
                    EdgeRecord(
                        i=i,
                        j=j,
                        motion_nll=float(motion_nll),
                        appearance_nll=float(appearance_nll),
                        gap_nll=float(gap_nll),
                        total=float(total),
                        gap_frames=int(gap_frames),
                        # KF end state for A to drive visualization later
                        kf_end=kf_cache[i],
                        # References to tracks for bbox/frame info
                        A_ref=A,
                        B_ref=B,
                    )
                )

        return graph

    # ──────────────────────────────── scheduling/splitting ─────────────────────────── #

    def _scheduled_start_cost(self, iteration: int) -> float:
        """Compute the start-cost for a given iteration based on configured schedule."""
        w = self.w_start_initial
        if self.start_cost_mode == 'geo':
            w = w * (self.start_cost_growth**iteration)
        else:
            # linear by default
            w = w + iteration * self.start_cost_growth
        if self.start_cost_max is not None:
            w = min(w, self.start_cost_max)
        return w

    def _maybe_split_tracks(
        self,
        tracks: List[Track],
        cmc: CMC,
    ) -> List[Track]:
        if not self.enable_splitting:
            return tracks

        out: List[Track] = []
        for track in tracks:
            out.extend(self._split_track(track, cmc))
        return out

    def _split_track(self, track: Track, cmc: CMC) -> List[Track]:
        """
        Split tracks when a per-step motion or appearance anomaly is detected.
        Motion: advance KF from previous detection to current and compute Mahalanobis distance → P_same.
        Appearance: compare current embedding to rolling mean of previous W embeddings.
        """

        detections = track.sorted_detections
        if len(detections) <= 1:
            return [track]

        G = self.internal_split_gap_frames if self.internal_split_gap_frames > 0 else float('inf')

        splits: List[Track] = []

        # initialize state with first detection
        state = KFState.init(detections[0])
        current_track = Track(track_id=track.track_id, sorted_detections=[detections[0]])

        for i in range(1, len(detections)):
            detection = detections[i]

            rest_track = Track(track_id=track.track_id, sorted_detections=detections[i:])
            # Motion anomaly
            motion_nll = _motion_nll(state, rest_track, cmc, self.max_detections_to_compare, self.use_position_only)
            appearance_nll = _appearance_nll(current_track, rest_track, self.appearance_similarity_gamma)

            # Split when evidence suggests different identity (high NLL = low P_same)
            split_due_to_motion = motion_nll > self.motion_split_nll_max
            split_due_to_app = appearance_nll > self.appearance_split_nll_max
            split_due_to_gap = detection.frame_idx - current_track.end_frame - 1 > G

            should_split = split_due_to_motion or split_due_to_app or split_due_to_gap
            allow_more_splits = (
                len(splits) < self.max_splits_per_track if self.max_splits_per_track is not None else True
            )

            if should_split and allow_more_splits:
                splits.append(current_track)
                current_track = Track(track_id=track.track_id, sorted_detections=[detection])
                state = KFState.init(detection)
            else:
                current_track.sorted_detections.append(detection)
                state = state.update_to_det(detection, cmc)  # incorporate measurement

        if current_track.sorted_detections:
            splits.append(current_track)

        return splits

    def _log(self, *args, **kwargs) -> None:
        if self.enable_edge_logging:
            print(*args, **kwargs)

    def _log_tracks(self, iteration: int, tracks: List[Track], message: str) -> None:
        if self.enable_edge_logging:
            print(f'Iteration {iteration}: {message}')
            for track in tracks:
                print(
                    f'  track[{track.track_id}] start={track.start_frame} end={track.end_frame} len={len(track.sorted_detections)}'
                )

    def _debug_display(self, debug: DebugSession, graph: FragmentGraph, iteration: int, cmc: CMC) -> None:
        if self.enable_edge_logging:
            # debug graph - print: count of edges, min/median/max of costs, and how many edges pass the scheduled cap
            _dbg_w = self._scheduled_start_cost(iteration)
            self._log(
                f'Iteration {iteration}: graph has {len(graph.pair_costs)} edges, min/median/max of costs: {min(graph.pair_costs.values())}/{np.median(list(graph.pair_costs.values()))}/{max(graph.pair_costs.values())}, {sum(1 for cost in graph.pair_costs.values() if cost < _dbg_w)} edges have cost < {_dbg_w}'
            )

        # Optional interactive graph visualization (source/sink + edges with costs)
        if self.enable_graph_viz:

            def on_edge_click(edge: EdgeRecord) -> None:
                if edge.A_ref is None or edge.B_ref is None or edge.kf_end is None:
                    return
                # Fetch required frames
                frame_a = debug.get_frame(edge.A_ref.end_frame)
                frame_b = debug.get_frame(edge.B_ref.start_frame)
                if frame_a is None or frame_b is None:
                    return
                # Build metrics
                metrics = EdgeMetrics(
                    motion_nll=float(edge.motion_nll),
                    appearance_nll=float(edge.appearance_nll),
                    gap_nll=float(edge.gap_nll),
                    total_cost=float(edge.total),
                    average_mahalanobis_squared=0.0,
                    average_log_determinant=0.0,
                    gap_frames=int(edge.gap_frames),
                )
                # Create a lightweight KFState-like container for end state
                composed = compose_fragment_pair_view(
                    edge.A_ref,
                    edge.B_ref,
                    metrics,
                    edge.kf_end,
                    cmc,
                    frame_a,
                    frame_b,
                )
                debug.show(composed)
                # Wait for user to proceed; Left/comma would be treated as -1 but no back-navigation here
                _ = debug.wait_step()

            def on_node_click(track: Track) -> None:
                f = debug.get_frame(int(track.start_frame))
                if f is None:
                    return
                debug.show(f, window_name=f'node-{track.track_id}')

            show_graph_interactive(
                graph.fragments,
                self._edge_debug_records,
                on_edge_click=on_edge_click,
                on_node_click=on_node_click,
            )
            # After closing the graph, optionally wait for a keypress to step iteration
            _ = debug.wait_step()


# ──────────────────────────────── cost helpers ─────────────────────────────── #


def clamp_prob(p: float) -> float:
    return max(EPS, min(1.0 - EPS, float(p)))


def NLL_from_prob(p: float) -> float:
    """Negative log-likelihood ratio cost: -logit(p)."""
    p = clamp_prob(p)
    return float(-math.log(p / (1.0 - p)))


def _appearance_nll(a: Track, b: Track, gamma: float) -> float:
    """Lab χ² distance between fragment prototypes → Platt → NLLR."""
    a_mean = a.mean_embedding()
    b_mean = b.mean_embedding()
    return NLL_from_prob(a_mean.probability(b_mean, gamma))


def _motion_nll(A: KFState, B: Track, cmc: CMC, max_detections_to_compare: int, use_position_only: bool) -> float:
    """
    Motion NLL between frags[idx_a] → frags[idx_b]:
      - use cached KF end state for A
      - predict by Δ to each of first K detections of B
      - inverse-GMC the observation into A.end frame
      - 0.5*d2 + 0.5*log|S_pos|, averaged across used dets
    Returns: (motion_nll)
    """
    nll_values: List[float] = []

    pred: KFState = A

    # evaluate up to K detections at B's start
    for detection in B.sorted_detections[:max_detections_to_compare]:
        # predict from A.end by Δ
        pred = pred.predict_to(detection.frame_idx, cmc)

        # position-only mahalanobis + log|S_pos|
        d2 = pred.gating_distance(detection.bbox.center_wh, only_position=use_position_only)

        p_same = probability_from_dist(d2, df=2 if use_position_only else 4)
        nll_values.append(NLL_from_prob(p_same))

    return float(np.mean(nll_values or [0.0]))


# ───────────────────────────────────── gap ─────────────────────────────────── #


def _gap_nll(gap_frames: int, p_miss: float) -> float:
    return float(gap_frames) * (-math.log(p_miss))


# ────────────────────────────── simple track splitter ───────────────────────── #


def _possible_mergeable_candidates(fragments: List[Track], max_frame_gap: int) -> List[Tuple[TrackId, TrackId]]:
    """Return possible mergeable candidates for a list of fragments."""
    candidates: List[Tuple[TrackId, TrackId]] = []
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            gap = fragments[i].frame_gap(fragments[j])
            if gap >= 0 and gap <= max_frame_gap:
                candidates.append((i, j))
    return candidates


def _are_all_tracks_merged(new_tracks: List[Track], max_frame_gap: int) -> bool:
    """Check if all tracks are merged."""
    return all(
        new_tracks[i].track_id == new_tracks[j].track_id
        for i, j in _possible_mergeable_candidates(new_tracks, max_frame_gap)
    )


def _are_assignments_the_same(last_tracks: List[Track], new_tracks: List[Track]) -> bool:
    """Check if the assignments are the same."""
    first_detections = {track.sorted_detections[0]: track for track in last_tracks}
    for track in new_tracks:
        if track.sorted_detections[0] not in first_detections:
            return False
        old_track = first_detections[track.sorted_detections[0]]
        if len(old_track.sorted_detections) != len(track.sorted_detections):
            return False
        for old_det, new_det in zip(old_track.sorted_detections, track.sorted_detections):
            if old_det != new_det:
                return False
    return True


def _remerge_tracks(tracks: List[Track]) -> List[Track]:
    """Re-merge tracks with the same track id."""
    merged_tracks: dict[TrackId, list[Detection]] = {}
    for track in tracks:
        merged_tracks.setdefault(track.track_id, []).extend(track.sorted_detections)
    return [
        Track(track_id=track_id, sorted_detections=sorted(detections, key=lambda d: d.frame_idx))
        for track_id, detections in merged_tracks.items()
    ]
