from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import math
import numpy as np

from server.inference.bot_sort.cmc import CMC
from server.inference.src.util.algebra import probability_from_dist


from ..util.video_io import VideoInfo, VideoReader
from ..common_types import Detection, Track, TrackId
from ..settings import MAX_OVERLAP_LENGTH_SECONDS, OPTIMIZER_W_START, EPS
from .ILP_graph_solver import FragmentGraph, ILPGraphSolver
from server.inference.bot_sort.kalman_filter import KFState, _KalmanFilter
from server.inference.src.visualization.stabilize import Transform
from server.inference.src.tracking.reid import HistogramEmbedding


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
        video_path: str,
        w_start: float = OPTIMIZER_W_START * 50,
        # costs
        p_miss: float = 0.97,  # for gap NLL
        # motion eval
        max_detections_to_compare: int = 2,  # eval first K detections of B (1..3 recommended)
        use_position_only: bool = True,  # gating_distance on (cx,cy) or (cx,cy,w,h)
        # iteration
        max_optimization_iterations: int = 4,
        # optional internal long-gap split during iteration (simple rule)
        internal_split_gap_frames: int = 3,  # 0 disables; >0 splits tracks on internal gaps > this
    ) -> None:
        self.video_path = video_path
        self.w_start = float(w_start)
        self.p_miss = float(p_miss)
        self.max_detections_to_compare = int(max(1, max_detections_to_compare))
        self.use_position_only = bool(use_position_only)
        self.max_optimization_iterations = int(max(1, max_optimization_iterations))
        self.internal_split_gap_frames = int(max(0, internal_split_gap_frames))

        # TODO remove
        # Debugging/visualization
        self.enable_edge_logging = False
        self.enable_graph_viz = False
        self.inline_edge_windows = False

        # Storage for debug visualization
        self._edge_debug_records: List[dict] = []
        self._edge_fragments: List[Track] = []

    # ───────────────────────────────── public API ───────────────────────────────── #

    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        """Run iterative graph building and ILP solve. Stops when cost does not improve."""

        cmc = CMC(transforms)
        max_frame_gap = int(round(MAX_OVERLAP_LENGTH_SECONDS * video_properties.fps))

        tracks = _maybe_split_on_internal_gaps(tracks, self.internal_split_gap_frames)
        best_cost = float('inf')

        frame_dict = {}
        with VideoReader(self.video_path) as reader:
            for frame_idx, frame in reader.read_frames():
                frame_dict[frame_idx] = frame

        for iteration in range(self.max_optimization_iterations):
            # Iteration-level debug: list fragments/ids before building the graph
            if self.enable_edge_logging:
                try:
                    print(f'Iteration {iteration}: fragments before graph build:')
                    for idx, tr in enumerate(sorted(tracks, key=lambda t: t.start_frame)):
                        print(
                            f'  it={iteration} frag[{idx}] id={tr.track_id} start={tr.start_frame} end={tr.end_frame} len={len(tr.sorted_detections)}'
                        )
                except Exception:
                    pass

            graph = self._build_fragment_graph(tracks, max_frame_gap, cmc)

            if self.enable_edge_logging:
                # debug graph - print:count of edges, min/median/max of costs, and how many edges have cost < current w_start*(it+1)
                print(
                    f'Iteration {iteration}: graph has {len(graph.pair_costs)} edges, min/median/max of costs: {min(graph.pair_costs.values())}/{np.median(list(graph.pair_costs.values()))}/{max(graph.pair_costs.values())}, {sum(1 for cost in graph.pair_costs.values() if cost < self.w_start * (iteration + 1))} edges have cost < {self.w_start * (iteration + 1)}'
                )

            # Optional interactive graph visualization (source/sink + edges with costs)
            if self.enable_edge_logging and self.enable_graph_viz:
                try:
                    self._show_graph_interactive(self._edge_fragments, self._edge_debug_records, cmc, frame_dict)
                except Exception:
                    # Non-fatal if viz fails (e.g., headless or missing deps)
                    pass
            # TODO increase start cost iteratively
            new_tracks, new_cost = ILPGraphSolver(self.w_start * (iteration + 1)).optimize_graph(graph)

            if self.enable_edge_logging:
                print(f'Iteration {iteration}: new tracks after ILP solve:')
                for track in new_tracks:
                    print(
                        f'  it={iteration} track[{track.track_id}] start={track.start_frame} end={track.end_frame} len={len(track.sorted_detections)}'
                    )

            if new_cost >= best_cost:  # no improvement → stop
                if self.enable_edge_logging:
                    print(f'No improvement in iteration {iteration}, stopping. {new_cost} >= {best_cost}')
                break
            tracks, best_cost = new_tracks, new_cost

            are_all_tracks_merged = all(
                new_tracks[i].track_id == new_tracks[j].track_id
                # Only tracks which were split on internal gaps remain, no other candidates
                for i, j in _possible_mergeable_candidates(new_tracks, max_frame_gap)
            )
            # TODO smarter stopping condition (no assignment changes?)
            if are_all_tracks_merged:
                if self.enable_edge_logging:
                    print(f'All tracks merged, stopping. {new_cost}')
                break

            # optional: after each solve, split again on internal gaps if enabled
            tracks = _maybe_split_on_internal_gaps(tracks, self.internal_split_gap_frames)

        # Remerge tracks with same track id in case of internal splits
        return _remerge_tracks(tracks)

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

        if self.enable_edge_logging:
            self._edge_debug_records = []

        for i, j in _possible_mergeable_candidates(fragments, max_frame_gap):
            A = fragments[i]
            B = fragments[j]
            # compute costs
            motion_nll, avg_d2, avg_logdet, used_k = _motion_nll(
                kf_cache[i],
                B,
                cmc,
                self.max_detections_to_compare,
                self.use_position_only,
            )
            if math.isinf(motion_nll) or math.isnan(motion_nll):
                continue

            appearance_nll = _appearance_nll(A, B)
            if math.isinf(appearance_nll) or math.isnan(appearance_nll):
                continue

            gap_frames = B.start_frame - A.end_frame - 1
            gap_nll = _gap_nll(gap_frames, self.p_miss)

            total = motion_nll + appearance_nll + gap_nll

            # Collect per-edge debug data for interactive visualization
            if self.enable_edge_logging:
                try:
                    cache_A = kf_cache[i]
                    self._edge_debug_records.append(
                        {
                            'i': i,
                            'j': j,
                            'track_id_a': A.track_id,
                            'track_id_b': B.track_id,
                            'start_a': A.start_frame,
                            'end_a': A.end_frame,
                            'start_b': B.start_frame,
                            'end_b': B.end_frame,
                            'motion_nll': float(motion_nll),
                            'appearance_nll': float(appearance_nll),
                            'gap_nll': float(gap_nll),
                            'total': float(total),
                            'avg_d2': float(avg_d2 if avg_d2 is not None else 0.0),
                            'avg_logdet': float(avg_logdet),
                            'used_k': int(used_k),
                            'gap_frames': int(gap_frames),
                            # KF end state for A to drive visualization later
                            'kf_mean_end': cache_A.mean,
                            'kf_cov_end': cache_A.cov,
                            'kf_end_frame': cache_A.last_frame,
                            # References to tracks for bbox/frame info
                            'A_ref': A,
                            'B_ref': B,
                        }
                    )
                except Exception:
                    pass

            graph.add_connection(i, j, float(total))

        # Store fragments and per-edge records for later interactive visualization
        if self.enable_edge_logging:
            self._edge_fragments = fragments

        return graph

    # ─────────────────────────────── debug visualization ─────────────────────────── #

    def _show_edge_debug(
        self,
        A: Track,
        B: Track,
        motion_nll: float,
        avg_d2: float,
        avg_logdet: float,
        appearance_nll: float,
        gap_nll: float,
        total_cost: float,
        frame_dict: Dict[int, np.ndarray],
        kf_mean_end: np.ndarray,
        kf_cov_end: np.ndarray,
        kf_end_frame: int,
        cmc: CMC,
    ) -> None:
        try:
            import cv2  # type: ignore
        except Exception:
            return

        try:
            frame_a = frame_dict.get(A.end_frame)
            frame_b = frame_dict.get(B.start_frame)
            if frame_a is None or frame_b is None:
                return

            vis_a = frame_a.copy()
            vis_b = frame_b.copy()

            # Draw detection bounding boxes
            bb_a = A.end.bbox
            bb_b = B.start.bbox
            cv2.rectangle(vis_a, (int(bb_a.x1), int(bb_a.y1)), (int(bb_a.x2), int(bb_a.y2)), (0, 255, 0), 2)
            cv2.rectangle(vis_b, (int(bb_b.x1), int(bb_b.y1)), (int(bb_b.x2), int(bb_b.y2)), (0, 255, 255), 2)

            # Labels near boxes
            cv2.putText(
                vis_a,
                f'A id={A.track_id} end f={A.end_frame}',
                (max(0, int(bb_a.x1)), max(0, int(bb_a.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis_b,
                f'B id={B.track_id} start f={B.start_frame}',
                (max(0, int(bb_b.x1)), max(0, int(bb_b.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Prepare KF instance once for overlays
            kf = _KalmanFilter()

            # Overlay KF current on A.end frame (mean_end, cov_end)
            try:
                cx_a, cy_a, w_a, h_a = kf.display_bbox(kf_mean_end, kf_cov_end, alpha=0.0)
                x1a = int(cx_a - w_a / 2.0)
                y1a = int(cy_a - h_a / 2.0)
                x2a = int(cx_a + w_a / 2.0)
                y2a = int(cy_a + h_a / 2.0)
                cv2.rectangle(vis_a, (x1a, y1a), (x2a, y2a), (255, 0, 0), 2)
                # Draw velocity arrow from KF state if available
                if kf_mean_end.shape[0] >= 6:
                    vx = float(kf_mean_end[4])
                    vy = float(kf_mean_end[5])
                    start_pt = (int(cx_a), int(cy_a))
                    end_pt = (int(round(cx_a + vx)), int(round(cy_a + vy)))
                    cv2.arrowedLine(vis_a, start_pt, end_pt, (255, 0, 0), 2, tipLength=0.3)
                cv2.putText(
                    vis_a,
                    'KF A end',
                    (x1a, max(0, y1a - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

            # Predict KF state from A.end to B.start and overlay on B.start frame
            try:
                m_b, P_b = kf.advance_state_to_frame(kf_mean_end, kf_cov_end, cmc, kf_end_frame, B.start_frame)
                cx_b, cy_b, w_b, h_b = kf.display_bbox(m_b, P_b, alpha=0.0)
                x1b = int(cx_b - w_b / 2.0)
                y1b = int(cy_b - h_b / 2.0)
                x2b = int(cx_b + w_b / 2.0)
                y2b = int(cy_b + h_b / 2.0)
                cv2.rectangle(vis_b, (x1b, y1b), (x2b, y2b), (255, 0, 255), 2)
                # Gating distance between predicted KF and B.start detection
                z = B.start.bbox.center_wh.reshape(1, 4).astype(np.float64)
                g2 = float(kf.gating_distance(m_b, P_b, z, only_position=self.use_position_only, metric='maha')[0])
                # Draw connection and label
                cv2.putText(
                    vis_b,
                    f'KF→B pred g^2={g2:.2f}',
                    (x1b, max(0, y1b - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                # Line between predicted KF center and detection center on B frame
                dcx = int((bb_b.x1 + bb_b.x2) / 2)
                dcy = int((bb_b.y1 + bb_b.y2) / 2)
                cv2.line(vis_b, (int(cx_b), int(cy_b)), (dcx, dcy), (0, 200, 255), 1)
                midx = int((cx_b + dcx) / 2)
                midy = int((cy_b + dcy) / 2)
                cv2.putText(
                    vis_b,
                    f'g^2={g2:.2f}',
                    (midx + 4, midy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

            # Normalize heights for side-by-side display
            ha, wa = vis_a.shape[:2]
            hb, wb = vis_b.shape[:2]
            target_h = max(ha, hb)
            scale_a = target_h / float(ha) if ha > 0 else 1.0
            scale_b = target_h / float(hb) if hb > 0 else 1.0
            vis_a_resized = cv2.resize(vis_a, (int(round(wa * scale_a)), target_h), interpolation=cv2.INTER_AREA)
            vis_b_resized = cv2.resize(vis_b, (int(round(wb * scale_b)), target_h), interpolation=cv2.INTER_AREA)
            combined = np.concatenate([vis_a_resized, vis_b_resized], axis=1)

            # Add a top banner with cost terms and gap frames
            banner_h = 36
            banner = np.full((banner_h, combined.shape[1], 3), 15, dtype=np.uint8)
            gap_frames = max(0, int(B.start_frame - A.end_frame - 1))
            text = (
                f'Δf={gap_frames} avg_d2={avg_d2:.3f} avg_logdet={avg_logdet:.3f} '
                f'C_mot={motion_nll:.4f}  C_app={appearance_nll:.4f}  C_gap={gap_nll:.4f}  C_tot={total_cost:.4f}'
            )
            cv2.putText(banner, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            canvas = np.concatenate([banner, combined], axis=0)
            canvas = cv2.resize(canvas, (canvas.shape[1] // 2, canvas.shape[0] // 2))

            window_name = 'ILP edge debug (A end | B start)'
            cv2.imshow(window_name, canvas)
            # Short wait to refresh window without blocking the entire optimization
            cv2.waitKey(0)
        except Exception:
            # Fail silently in debug drawing
            return

    # ─────────────────────────────── interactive graph viz ────────────────────────── #

    def _show_graph_interactive(
        self,
        fragments: List[Track],
        edge_records: List[dict],
        cmc: CMC,
        frame_dict: Dict[int, np.ndarray],
    ) -> None:
        """Render an interactive directed graph of fragments with source/sink.
        Clicking an edge displays the existing per-edge OpenCV debug view.
        Clicking a node displays a single-frame bbox annotation for that fragment.
        """
        if not edge_records:
            return
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.patches as mpatches  # type: ignore
        import networkx as nx  # type: ignore

        # Build a DiGraph for visualization
        G = nx.DiGraph()

        # Nodes for fragments
        for idx, frag in enumerate(fragments):
            label = f'{idx} (id={frag.track_id})\n[{frag.start_frame}-{frag.end_frame}]'
            G.add_node(idx, label=label, start=frag.start_frame, end=frag.end_frame)

        # Add edges from records
        for rec in edge_records:
            i = rec['i']
            j = rec['j']
            G.add_edge(i, j, total=float(rec['total']))

        # Source/sink nodes
        SRC = 'SOURCE'
        SNK = 'SINK'
        G.add_node(SRC)
        G.add_node(SNK)

        all_nodes = [n for n in G.nodes if isinstance(n, int)]
        indeg = {n: 0 for n in all_nodes}
        outdeg = {n: 0 for n in all_nodes}
        for u, v in [(rec['i'], rec['j']) for rec in edge_records]:
            outdeg[u] += 1
            indeg[v] += 1
        for n in all_nodes:
            if indeg[n] == 0:
                G.add_edge(SRC, n, total=0.0)
            if outdeg[n] == 0:
                G.add_edge(n, SNK, total=0.0)

        # Positions: left-to-right by time, Y randomized per node with stable seed
        min_start = min(f.start_frame for f in fragments) if fragments else 0
        max_end = max(f.end_frame for f in fragments) if fragments else 1
        span = max(1, max_end - min_start)

        pos: Dict[object, tuple] = {}
        for idx, frag in enumerate(fragments):
            x = (frag.start_frame - min_start) / span
            # Stable pseudo-random Y in (0.08, 0.92)
            rnd = float(np.random.RandomState(idx * 9973 + 811).rand())
            y = 0.08 + 0.84 * rnd
            pos[idx] = (x, y)
        # Source/Sink just outside range
        pos[SRC] = (-0.08, 0.5)
        pos[SNK] = (1.08, 0.5)

        fig, ax = plt.subplots(figsize=(max(6, len(fragments) * 0.9), 6))
        ax.set_title('ILP Fragment Graph (click node for bbox, edge for comparison)')
        ax.set_axis_off()

        # Draw nodes
        node_labels = {n: (G.nodes[n]['label'] if isinstance(n, int) else n) for n in G.nodes}
        int_nodes = [n for n in G.nodes if isinstance(n, int)]
        nodes_main = nx.draw_networkx_nodes(G, pos, nodelist=int_nodes, node_color='#99c2ff', node_size=800, ax=ax)
        try:
            nodes_main.set_picker(True)  # make node markers pickable
        except Exception:
            pass
        nx.draw_networkx_nodes(G, pos, nodelist=[SRC, SNK], node_color='#dddddd', node_size=900, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)

        # Draw edges individually with pickable artists and cost labels
        artist_to_record: Dict[object, dict] = {}
        for rec in edge_records:
            i = rec['i']
            j = rec['j']
            p0 = pos[i]
            p1 = pos[j]
            arrow = mpatches.FancyArrowPatch(
                p0,
                p1,
                arrowstyle='-|>',
                mutation_scale=12,
                color='#444444',
                linewidth=1.2,
                alpha=0.9,
            )
            arrow.set_picker(True)
            ax.add_patch(arrow)
            artist_to_record[arrow] = rec

            # Cost label at mid-point
            mx = (p0[0] + p1[0]) / 2.0
            my = (p0[1] + p1[1]) / 2.0
            txt = ax.text(
                mx,
                my,
                f'{rec["total"]:.2f}',
                fontsize=8,
                color='#222222',
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7),
            )
            txt.set_picker(True)
            artist_to_record[txt] = rec

        # Draw SRC/SNK edges (not pickable)
        for u, v in G.edges:
            if isinstance(u, str) or isinstance(v, str):
                p0 = pos[u]
                p1 = pos[v]
                ax.add_patch(
                    mpatches.FancyArrowPatch(
                        p0, p1, arrowstyle='-|>', mutation_scale=12, color='#bbbbbb', linewidth=1.0, alpha=0.8
                    )
                )

        # Map PathCollection indices back to node ids for node click handling
        node_index_to_id = {i: node_id for i, node_id in enumerate(int_nodes)}

        def on_pick(event):
            artist = event.artist
            # Node clicks (PathCollection of main nodes)
            if artist is nodes_main:
                try:
                    ind_list = getattr(event, 'ind', None)
                    if not ind_list:
                        return
                    # Take the first picked index
                    pick_idx = int(ind_list[0])
                    node_id = node_index_to_id.get(pick_idx)
                    if node_id is None:
                        return
                    frag = fragments[node_id]
                    self._show_node_debug(frag, frame_dict)
                except Exception:
                    pass
                return

            # Edge clicks (arrows or cost labels)
            rec = artist_to_record.get(artist)
            if not rec:
                return
            try:
                self._show_edge_debug(
                    rec['A_ref'],
                    rec['B_ref'],
                    rec['motion_nll'],
                    rec['avg_d2'],
                    rec['avg_logdet'],
                    rec['appearance_nll'],
                    rec['gap_nll'],
                    rec['total'],
                    frame_dict,
                    rec['kf_mean_end'],
                    rec['kf_cov_end'],
                    rec['kf_end_frame'],
                    cmc,
                )
            except Exception:
                pass

        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            # Headless env: ignore
            pass

    def _show_node_debug(self, frag: Track, frame_dict: Dict[int, np.ndarray]) -> None:
        """Display a single frame for a fragment with its bbox annotated.
        Uses the fragment's first detection frame.
        """
        try:
            import cv2  # type: ignore
        except Exception:
            return
        try:
            if not frag.sorted_detections:
                return
            d = frag.sorted_detections[0]
            frame = frame_dict.get(d.frame_idx)
            if frame is None:
                return
            vis = frame.copy()
            bb = d.bbox
            cv2.rectangle(vis, (int(bb.x1), int(bb.y1)), (int(bb.x2), int(bb.y2)), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f'id={frag.track_id} f={d.frame_idx}',
                (max(0, int(bb.x1)), max(0, int(bb.y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow('ILP node debug (fragment start)', vis)
            cv2.waitKey(0)
        except Exception:
            return


# ──────────────────────────────── cost helpers ─────────────────────────────── #


def clamp_prob(p: float) -> float:
    return max(EPS, min(1.0 - EPS, float(p)))


def NLL_from_prob(p: float) -> float:
    """Negative log-likelihood ratio cost: -logit(p)."""
    p = clamp_prob(p)
    return float(-math.log(p / (1.0 - p)))


def _appearance_nll(a: Track, b: Track) -> float:
    """Lab χ² distance between fragment prototypes → Platt → NLLR."""
    a_mean = a.mean_embedding()
    b_mean = b.mean_embedding()
    assert isinstance(a_mean, HistogramEmbedding), f'assuming histogram embedding but got {type(a_mean)}'
    assert isinstance(b_mean, HistogramEmbedding), f'assuming histogram embedding but got {type(b_mean)}'
    probability = a_mean.probability(b_mean)
    return NLL_from_prob(probability)


def _motion_nll(
    A: KFState,
    B: Track,
    cmc: CMC,
    max_detections_to_compare: int,
    use_position_only: bool,
) -> Tuple[float, Optional[float], float, int]:
    """
    Motion NLL between frags[idx_a] → frags[idx_b]:
      - use cached KF end state for A
      - predict by Δ to each of first K detections of B
      - inverse-GMC the observation into A.end frame
      - 0.5*d2 + 0.5*log|S_pos|, averaged across used dets
    Returns: (motion_nll, avg_d2, logdet, used_K)
    """
    if not B.sorted_detections:
        return 0.0, 0.0, 0.0, 0

    total = 0.0
    d2_vals: List[float] = []
    logdet_vals: List[float] = []
    used = 0

    cost_vals: List[float] = []

    pred: KFState = A

    # evaluate up to K detections at B's start
    for db in B.sorted_detections[:max_detections_to_compare]:
        # predict from A.end by Δ
        pred = pred.predict_to(db.frame_idx, cmc)

        z_obs_back = db.bbox.center_wh

        # position-only mahalanobis + log|S_pos|
        d2 = pred.gating_distance(z_obs_back, only_position=use_position_only)
        logdet = pred.logdet(use_position_only)

        total += 0.5 * d2 + 0.5 * logdet
        # d2_vals.append(d2)
        logdet_vals.append(logdet)
        used += 1

        p_same = probability_from_dist(d2, df=2 if use_position_only else 4)
        d2_vals.append(p_same)
        cost_vals.append(NLL_from_prob(p_same))

    if used == 0:
        return 1e6, None, 0.0, 0

    avg_d2 = float(np.mean(d2_vals))
    avg_logdet = float(np.mean(logdet_vals))
    motion_nll = total / used
    motion_nll = float(np.mean(cost_vals))
    return motion_nll, avg_d2, avg_logdet, used


# ───────────────────────────────────── gap ─────────────────────────────────── #


def _gap_nll(gap_frames: int, p_miss: float) -> float:
    return float(gap_frames) * (-math.log(p_miss))


# ────────────────────────────── simple track splitter ───────────────────────── #


def _maybe_split_on_internal_gaps(tracks: List[Track], internal_split_gap_frames: int) -> List[Track]:
    """
    Optional conservative splitter: if enabled, breaks tracks at internal gaps
    > `internal_split_gap_frames`. Keeps the same track_id for resulting fragments.
    If disabled, returns input unchanged.
    """
    G = internal_split_gap_frames
    if G <= 0:  # Skip splitting - disabled # TODO what about other parameters for KF or Appearance uncertainty?
        return tracks

    out: List[Track] = []
    for tr in tracks:
        if len(tr.sorted_detections) <= 1:
            out.append(tr)
            continue
        run: List[Detection] = []
        last_f = None
        for d in tr.sorted_detections:
            if last_f is None or (d.frame_idx - last_f) <= G:  # TODO also split on KF or Appearance uncertainty
                run.append(d)
            else:
                # TODO does that work with multiple tracks with the same track_id?
                out.append(Track(track_id=tr.track_id, sorted_detections=run))
                run = [d]
            last_f = d.frame_idx
        if run:
            out.append(Track(track_id=tr.track_id, sorted_detections=run))
    return out


def _possible_mergeable_candidates(fragments: List[Track], max_frame_gap: int) -> List[Tuple[TrackId, TrackId]]:
    """Return possible mergeable candidates for a list of fragments."""
    candidates: List[Tuple[TrackId, TrackId]] = []
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            gap = fragments[j].start_frame - fragments[i].end_frame - 1
            if gap >= 0 and gap <= max_frame_gap:
                candidates.append((i, j))
    return candidates


def _remerge_tracks(tracks: List[Track]) -> List[Track]:
    """Re-merge tracks with the same track id."""
    merged_tracks: dict[TrackId, list[Detection]] = {}
    for track in tracks:
        merged_tracks.setdefault(track.track_id, []).extend(track.sorted_detections)
    return [
        Track(track_id=track_id, sorted_detections=sorted(detections, key=lambda d: d.frame_idx))
        for track_id, detections in merged_tracks.items()
    ]
