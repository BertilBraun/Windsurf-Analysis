from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import math
import numpy as np

from server.inference.src.motion.cmc import CMC
from server.inference.src.util.algebra import NLL_from_prob, lerp, probability_from_dist
from server.inference.src.visualization.debug.session import DebugSession


from ..util.video_io import VideoInfo
from ..common_types import Point, Track, TrackId
from ..settings import MAX_OVERLAP_LENGTH_SECONDS
from .ILP_graph_solver import FragmentGraph, ILPGraphSolver
from server.inference.src.motion.kalman_filter import KFState
from server.inference.src.visualization.stabilize import Transform
from server.inference.src.visualization.debug import get_debug_session
from server.inference.src.visualization.debug.graph import EdgeRecord, show_graph_interactive
from server.inference.src.visualization.debug.draw import draw_bounding_box, draw_heatmap
from server.inference.src.visualization.debug.overlays import EdgeMetrics, compose_fragment_pair_view


class ILPTracker:
    """ILP tracker.

    Plan:
    1. Build a graph of possible fragment connections with their costs.
        - Possible connections (A -> B) are based on:
            - A.end_frame < B.start_frame and B.start_frame <= A.end_frame + MAX_OVERLAP_LENGTH_SECONDS * video_fps
        - Costs are based on:
            To calculate the actual cost, we use the sum of the NLL for motion, appearance and gap.
            - Motion: KF Mahalanobis NLL + GMC (position-only) + 0.5*log|S_pos|. We apply the appropriate Camera Transforms on each frame. Transfors are defined as: Transform = NamedTuple('Transform', [('dx', float), ('dy', float), ('da', float), ('frame_idx', int)]) # dx, dy, da for each frame relative to the previous frame.
            - Appearance: embedding is a LAB color histogram, we compute the mean histogram for both A and B and then use the chi-squared distance to get the appearance similarity probability by calculating platt_prob_from_dist. Lab χ² → Platt prob → NLLR
            - Gap: per-frame miss NLL
    2. Solve the ILP problem
    3. Return the solution.
    """

    def __init__(
        self,
        video_path: str | None = None,
        # Start-cost schedule (controls edge cap and start penalty in ILP)
        w_start: float = 90.19570414175368,  # initial start cost (backward-compatible)
        w_end: Optional[float] = None,  # initial end cost
        # Per-term cost weights
        w_motion: float = 0.3708452033348393,
        w_appearance: float = 2.2935669734544937,
        w_gap: float = 9.446696614284106,
        # Gap model
        p_miss: float = 0.9411498635286982,  # for gap NLL
        # Motion evaluation
        max_detections_to_compare: int = 2,  # eval first K detections of B (1..3 recommended)
        use_position_only: bool = True,  # gating_distance on (cx,cy) or (cx,cy,w,h)
        # Appearance similarity
        appearance_similarity_gamma: float = 11.535947876483421,
    ) -> None:
        self.video_path = video_path
        self.w_start = w_start
        self.w_end = w_end if w_end is not None else w_start
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

        # TODO remove
        # Debugging/visualization
        self.enable_edge_logging = False
        self.enable_graph_viz = False
        self.inline_edge_windows = False

        # Storage for debug visualization
        self._edge_debug_records: List[EdgeRecord] = []

    # ───────────────────────────────── public API ───────────────────────────────── #
    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        """Run graph building and ILP solve."""

        cmc = CMC(transforms)
        max_frame_gap = int(round(MAX_OVERLAP_LENGTH_SECONDS * video_properties.fps))

        # Initialize debug session if any debug feature is enabled
        debug_enabled = bool(self.enable_edge_logging or self.enable_graph_viz)
        debug = get_debug_session(self.video_path or '', enabled=debug_enabled)

        # Reset debug edge storage per run (prevents cross-run accumulation)
        self._edge_debug_records = []

        graph = self._build_fragment_graph(tracks, max_frame_gap, cmc)

        self._debug_display(debug, graph, 0, cmc)

        # Solve with scheduled start cost
        start_costs, end_costs = self._compute_spatial_costs(tracks, video_properties)

        # Pass max_cost_limit as the max of start and end costs to allow pruning
        max_cost_limit = max(self.w_start, self.w_end)

        new_tracks, new_cost = ILPGraphSolver().optimize_graph(graph, start_costs, end_costs, max_cost_limit)

        debug.close()

        # Remerge tracks with same track id in case of internal splits
        print(f'ILP tracker finished with {len(new_tracks)} tracks')
        return new_tracks

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

    def _compute_spatial_costs(
        self, tracks: List[Track], video_properties: VideoInfo
    ) -> Tuple[List[float], List[float]]:
        """Compute start and end costs for each track based on spatial position (border proximity).
        We want to encourage tracks to start and end near the border of the image, not in the image center."""
        start_costs = []
        end_costs = []
        W = video_properties.width
        H = video_properties.height
        margin = 0.1  # 20% of the width or height

        def interpolation_factor(center: Point) -> float:
            # interpolate from 1 all the way at the border to 0 at a box in the center of the image of size (1-margin*2)x(1-margin*2)
            nx, ny = center.x / W, center.y / H
            fx = max(0.0, margin - nx, nx - (1.0 - margin)) / margin
            fy = max(0.0, margin - ny, ny - (1.0 - margin)) / margin

            return min(1.0, max(fx, fy))

        # Lower cost for starting/ending near border
        # We use a fixed low cost for border starts/ends to encourage them
        low_cost_start = self.w_start / 2.0
        low_cost_end = self.w_end / 2.0

        for track in tracks:
            # Start cost
            factor = interpolation_factor(track.start.bbox.center)
            start_costs.append(lerp(self.w_start, low_cost_start, factor))

            # End cost
            factor = interpolation_factor(track.end.bbox.center)
            end_costs.append(lerp(self.w_end, low_cost_end, factor))

        return start_costs, end_costs

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
            _dbg_w = self.w_start
            self._log(
                f'Iteration {iteration}: graph has {len(graph.pair_costs)} edges, min/median/max of costs: {min(graph.pair_costs.values())}/{np.median(list(graph.pair_costs.values()))}/{max(graph.pair_costs.values())}, {sum(1 for cost in graph.pair_costs.values() if cost < _dbg_w)} edges have cost < {_dbg_w}'
            )

        # Optional interactive graph visualization (source/sink + edges with costs)
        if self.enable_graph_viz:

            def _show_node_candidate_heatmaps(*, node_idx: int, direction: str) -> None:
                """
                Show greedy-style heatmaps for all ILP candidate edges incident to this node.

                direction:
                  - 'out': node is edge.i (A -> B)
                  - 'in' : node is edge.j (A -> B)
                """
                assert direction in ('out', 'in')

                # Collect relevant edges
                if direction == 'out':
                    edges = [e for e in self._edge_debug_records if e.i == node_idx]
                    other_indices = [e.j for e in edges]
                    title_prefix = 'Outgoing candidates'
                else:
                    edges = [e for e in self._edge_debug_records if e.j == node_idx]
                    other_indices = [e.i for e in edges]
                    title_prefix = 'Incoming candidates'

                if not edges:
                    return

                # Column labels: other fragment's track_id (more meaningful than fragment index)
                col_labels: List[str] = []
                for other_idx in other_indices:
                    try:
                        col_labels.append(str(graph.fragments[other_idx].track_id))
                    except Exception:
                        col_labels.append(str(other_idx))

                # One-row matrices (like greedy stitcher's "tracks x detections" grid, but for one node)
                row_label = f'{direction}:{graph.fragments[node_idx].track_id}'
                # Convert costs back to probabilities for interpretation.
                # Motion/appearance use NLL_from_prob(p) = -logit(p) => p = 1 / (1 + exp(nll)).
                # Gap uses gap_nll = -log(p_miss^gap) => p_gap = exp(-gap_nll).
                motion_prob = np.array([[1.0 / (1.0 + math.exp(float(e.motion_nll))) for e in edges]], dtype=np.float32)
                appearance_prob = np.array(
                    [[1.0 / (1.0 + math.exp(float(e.appearance_nll))) for e in edges]], dtype=np.float32
                )
                gap_prob = np.array([[math.exp(-float(e.gap_nll)) for e in edges]], dtype=np.float32)

                hm_motion = draw_heatmap(
                    motion_prob,
                    row_labels=[row_label],
                    col_labels=col_labels,
                    title=f'{title_prefix}: motion probability (higher = better)',
                    vmin=0.0,
                    vmax=1.0,
                )
                hm_appearance = draw_heatmap(
                    appearance_prob,
                    row_labels=[row_label],
                    col_labels=col_labels,
                    title=f'{title_prefix}: appearance probability (higher = better)',
                    vmin=0.0,
                    vmax=1.0,
                )
                hm_gap = draw_heatmap(
                    gap_prob,
                    row_labels=[row_label],
                    col_labels=col_labels,
                    title=f'{title_prefix}: gap probability p_miss^gap (higher = better)',
                    vmin=0.0,
                    vmax=1.0,
                )

                # DebugSession.show() always composes side-by-side, so to stack vertically we pre-compose.
                stacked = np.concatenate([hm_motion, hm_appearance, hm_gap], axis=0)
                window_name = f'node-{graph.fragments[node_idx].track_id}-candidates-{direction}'
                debug.show(stacked, window_name=window_name)

                # Enable click-to-inspect: click a cell to open the *other* tracklet's start frame with bbox.
                # NOTE: draw_heatmap() uses these defaults.
                left_margin = 36
                top_margin = 50
                cell_width = 80
                cell_height = 80
                panel_h = int(hm_motion.shape[0])
                cols = len(edges)

                def on_heatmap_click(event: int, x: int, y: int, flags: int, param: object) -> None:
                    try:
                        import cv2  # local import; debug-only dependency
                    except Exception:
                        return
                    if event != cv2.EVENT_LBUTTONDOWN:
                        return
                    if cols <= 0:
                        return

                    panel_idx = int(y // max(1, panel_h))
                    if panel_idx < 0 or panel_idx > 2:
                        return
                    local_y = int(y - panel_idx * panel_h)

                    # Only accept clicks inside the matrix area (ignore title/labels/colorbar)
                    if x < left_margin or local_y < top_margin:
                        return
                    col = int((x - left_margin) // max(1, cell_width))
                    row = int((local_y - top_margin) // max(1, cell_height))
                    if row != 0 or col < 0 or col >= cols:
                        return

                    other_idx = other_indices[col]
                    other_track = graph.fragments[other_idx]
                    frame = debug.get_frame(int(other_track.start_frame))
                    if frame is None:
                        return
                    vis = frame.copy()
                    draw_bounding_box(
                        vis,
                        other_track.start.bbox,
                        (0, 255, 255),
                        label=f'track {other_track.track_id} start f={other_track.start_frame}',
                    )
                    debug.show(vis, window_name=f'inspect-tracklet-{other_track.track_id}')

                debug.set_mouse_callback(window_name, on_heatmap_click)

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
                # Find the fragment index for the clicked node (graph.fragments holds the same Track objects)
                try:
                    node_idx = graph.fragments.index(track)
                except ValueError:
                    node_idx = -1

                # Show the actual windsurfer (bbox) on the END frame for this node
                # (we usually want to link from this tracklet's end to future tracklets).
                f = debug.get_frame(int(track.end_frame))
                if f is None:
                    return
                vis = f.copy()
                draw_bounding_box(
                    vis,
                    track.end.bbox,
                    (0, 255, 255),
                    label=f'track {track.track_id} end f={track.end_frame}',
                )
                debug.show(vis, window_name=f'node-{track.track_id}')

                # Also show heatmaps for all candidates that can link to/from this node in the ILP graph
                if node_idx >= 0:
                    _show_node_candidate_heatmaps(node_idx=node_idx, direction='out')
                    _show_node_candidate_heatmaps(node_idx=node_idx, direction='in')

            show_graph_interactive(
                graph.fragments,
                self._edge_debug_records,
                on_edge_click=on_edge_click,
                on_node_click=on_node_click,
            )
            # After closing the graph, optionally wait for a keypress to step iteration
            _ = debug.wait_step()


# ──────────────────────────────── cost helpers ─────────────────────────────── #


def _appearance_nll(a: Track, b: Track, gamma: float) -> float:
    """Lab χ² distance between fragment prototypes → Platt → NLLR."""
    a_mean = a.mean_embedding(ema=0.6)
    b_mean = b.mean_embedding_reverse(ema=0.6)
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
