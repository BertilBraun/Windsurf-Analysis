from __future__ import annotations

"""Discrete‑optimization based multi‑object tracker using Z3.

This implementation now separates **local geometric continuity** from a **global
appearance cohesion** term:

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
    size or clip length. (We only create directed edges i→j for j>i to avoid
    double counting.)

    *K_prev* = *K_next* = 5 by default (configurable). Set them larger for more
    robust global cohesion; costs remain O(N * K) rather than O(N^2).

Other components preserved:
    • A decision variable per detection for its track id (domain 0‥n_tracks-1).
    • Frame‑level exclusivity (AllDifferent per frame).
    • Greedy pre‑processing creates *must‑link* groups (short obvious
      fragments) whose detections are forced to share a track id.

IMPORTANT ASSUMPTIONS
———————————————————
* Max per‑frame detection count ≤ n_tracks else UNSAT.
* Every detection is assigned to some track (no unassigned sentinel).
* Embeddings are L2‑normalised.
* Global embedding cost is quadratic in #detections (pairwise). If this becomes
  slow, restrict to a temporal window or subsample pairs.
"""

import z3
import logging
from typing import Dict, List, Tuple, Optional, Set

from video_io import VideoInfo
from common_types import Detection, Track, cosine_similarity
from tracking.greedy_tracker import GreedyTracker, _average_embedding

TIMEOUT_SECONDS = 60


class TimeoutException(Exception):
    """Raised when the Z3 solver times out."""


class UnsatisfiableException(Exception):
    """Raised when the Z3 solver finds the problem unsatisfiable."""


class FragmentGraph:
    """Represents the graph structure of fragment connections and their costs."""

    def __init__(self, fragments: List[Track]):
        self.fragments = fragments
        self.successors: List[List[int]] = [[] for _ in range(len(fragments))]
        self.pair_costs: Dict[Tuple[int, int], float] = {}

    def add_connection(self, from_idx: int, to_idx: int, cost: float) -> None:
        """Add a connection between two fragments with the given cost."""
        self.successors[from_idx].append(to_idx)
        self.pair_costs[(from_idx, to_idx)] = cost

    def get_outgoing_connections(self, fragment_idx: int) -> List[int]:
        """Get all outgoing connections for a fragment."""
        return self.successors[fragment_idx]

    def get_connection_cost(self, from_idx: int, to_idx: int) -> float:
        """Get the cost of a connection between two fragments."""
        return self.pair_costs[(from_idx, to_idx)]

    def get_all_connections(self) -> Dict[Tuple[int, int], float]:
        """Get all connections and their costs."""
        return self.pair_costs


class DiscreteOptimizationTracker:
    """Track objects by solving an assignment problem with *Z3*."""

    def __init__(
        self,
        max_link_gap: int = 25 * 5,
        min_link_iou: float = 0.0,
        min_link_cos: float = -1.0,
        w_link_iou: float = 0.2,
        w_link_app: float = 1.0,
        w_link_gap: float = 0.001,
        w_start: float = 10.0,  # <-- should be scaled according to number of estimated starts / tracks and number links required
        # the amount of frames to look forward and backwards for appearance.
        # For now these are not weighted by distance so keep small
        link_cost_appearance_window_radius: int = 10,
    ):
        # Fragment linking config
        self.max_link_gap = max_link_gap
        self.min_link_iou = min_link_iou
        self.min_link_cos = min_link_cos
        self.w_link_iou = w_link_iou
        self.w_link_app = w_link_app
        self.w_link_gap = w_link_gap
        self.w_start = w_start
        self.link_cost_appearance_window_radius = link_cost_appearance_window_radius

    def track_detections(self, detections: List[Detection], video_properties: VideoInfo) -> List[Track]:
        """Main entry point for tracking detections."""
        logging.info(f'{"=" * 80} Running discrete optimization tracker with {len(detections)} detections {"=" * 80}')

        fragments = self._create_initial_fragments(detections, video_properties)
        logging.info(f'{"=" * 80} Running discrete optimization tracker with {len(fragments)} fragments {"=" * 80}')

        return self._optimize_fragments(fragments)

    def _create_initial_fragments(self, detections: List[Detection], video_properties: VideoInfo) -> List[Track]:
        """Create initial fragments using greedy tracker."""
        return GreedyTracker().track_detections(detections, video_properties)

    def _optimize_fragments(self, fragments: List[Track]) -> List[Track]:
        """Optimize fragment connections using Z3 solver."""
        if not fragments:
            return []

        # Sort fragments by start frame
        fragments = sorted(fragments, key=lambda t: t.sorted_detections[0].frame_idx)

        # Build fragment connection graph
        graph = self._build_fragment_graph(fragments)

        # Solve optimization problem
        solution = self._solve_optimization_problem(graph)

        # Reconstruct tracks from solution
        return self._reconstruct_tracks_from_solution(fragments, solution)

    def _build_fragment_graph(self, fragments: List[Track]) -> FragmentGraph:
        """Build a graph of possible fragment connections with their costs."""
        graph = FragmentGraph(fragments)
        N = len(fragments)

        for i, start_fragment in enumerate(fragments):
            start_frames = self._get_fragment_frames(start_fragment)

            for j in range(i, N):
                end_fragment = fragments[j]
                end_frames = self._get_fragment_frames(end_fragment)

                # Skip if fragments have overlapping frames
                if start_frames.intersection(end_frames):
                    continue

                # Calculate connection cost
                cost = self._calculate_link_cost(start_fragment, end_fragment)
                if cost is not None:
                    graph.add_connection(i, j, cost)

        return graph

    def _get_fragment_frames(self, fragment: Track) -> Set[int]:
        """Get the set of frame indices for a fragment."""
        return {detection.frame_idx for detection in fragment.sorted_detections}

    def _solve_optimization_problem(self, graph: FragmentGraph) -> Dict[int, Optional[int]]:
        """Solve the fragment linking optimization problem using Z3."""
        opt = self._create_z3_optimizer()

        # Create decision variables
        link_vars = self._create_link_variables(graph)
        start_vars = self._create_start_variables(len(graph.fragments))

        # Add constraints
        self._add_outgoing_constraints(opt, graph, link_vars)
        self._add_incoming_constraints(opt, graph, link_vars)
        self._add_start_constraints(opt, graph, link_vars, start_vars)

        # Set objective function
        self._set_objective_function(opt, graph, link_vars, start_vars)

        # Solve and return solution
        return self._solve_and_extract_solution(opt, graph, link_vars)

    def _create_z3_optimizer(self) -> z3.Optimize:
        """Create and configure Z3 optimizer."""
        opt = z3.Optimize()
        opt.set('timeout', TIMEOUT_SECONDS * 1000)
        return opt

    def _create_link_variables(self, graph: FragmentGraph) -> Dict[Tuple[int, int], z3.BoolRef]:
        """Create boolean variables for fragment links."""
        return {(i, j): z3.Bool(f'link_{i}_{j}') for (i, j) in graph.get_all_connections()}

    def _create_start_variables(self, num_fragments: int) -> List[z3.BoolRef]:
        """Create boolean variables for fragment starts."""
        return [z3.Bool(f'start_{i}') for i in range(num_fragments)]

    def _add_outgoing_constraints(
        self, opt: z3.Optimize, graph: FragmentGraph, link_vars: Dict[Tuple[int, int], z3.BoolRef]
    ) -> None:
        """Add constraints ensuring each fragment has at most one outgoing link."""
        for i in range(len(graph.fragments)):
            outgoing_connections = graph.get_outgoing_connections(i)
            if outgoing_connections:
                out_links = [link_vars[(i, j)] for j in outgoing_connections]
                opt.add(z3.PbLe([(v, 1) for v in out_links], 1))

    def _add_incoming_constraints(
        self, opt: z3.Optimize, graph: FragmentGraph, link_vars: Dict[Tuple[int, int], z3.BoolRef]
    ) -> None:
        """Add constraints ensuring each fragment has at most one incoming link."""
        # Build incoming connections mapping
        incoming: List[List[z3.BoolRef]] = [[] for _ in range(len(graph.fragments))]
        for (i, j), var in link_vars.items():
            incoming[j].append(var)

        # Add constraints
        for j in range(len(graph.fragments)):
            if incoming[j]:
                opt.add(z3.PbLe([(v, 1) for v in incoming[j]], 1))

    def _add_start_constraints(
        self,
        opt: z3.Optimize,
        graph: FragmentGraph,
        link_vars: Dict[Tuple[int, int], z3.BoolRef],
        start_vars: List[z3.BoolRef],
    ) -> None:
        """Add constraints defining when a fragment is a start of a track."""
        # Build incoming connections mapping
        incoming: List[List[z3.BoolRef]] = [[] for _ in range(len(graph.fragments))]
        for (i, j), var in link_vars.items():
            incoming[j].append(var)

        # Add start constraints
        for i in range(len(graph.fragments)):
            if incoming[i]:
                opt.add(start_vars[i] == z3.And([z3.Not(v) for v in incoming[i]]))
            else:
                opt.add(start_vars[i])

    def _set_objective_function(
        self,
        opt: z3.Optimize,
        graph: FragmentGraph,
        link_vars: Dict[Tuple[int, int], z3.BoolRef],
        start_vars: List[z3.BoolRef],
    ) -> None:
        """Set the objective function to minimize total cost."""
        # Link costs
        link_cost_terms = [
            z3.If(var, z3.RealVal(graph.get_connection_cost(i, j)), z3.RealVal(0.0))
            for (i, j), var in link_vars.items()
        ]

        # Start costs
        start_cost_terms = [z3.If(var, z3.RealVal(self.w_start), z3.RealVal(0.0)) for var in start_vars]

        # Total cost
        all_terms = link_cost_terms + start_cost_terms
        total_cost = z3.Sum(all_terms) if all_terms else z3.RealVal(0.0)
        opt.minimize(total_cost)

    def _solve_and_extract_solution(
        self, opt: z3.Optimize, graph: FragmentGraph, link_vars: Dict[Tuple[int, int], z3.BoolRef]
    ) -> Dict[int, Optional[int]]:
        """Solve the optimization problem and extract the solution."""
        result = opt.check()

        if result != z3.sat:
            if result == z3.unknown:
                raise TimeoutException('Fragment linking solver timeout')
            if result == z3.unsat:
                raise UnsatisfiableException('Fragment linking UNSAT')
            raise RuntimeError(f'Unexpected solver status {result}')

        # Extract solution
        model = opt.model()
        successor_of: Dict[int, Optional[int]] = {i: None for i in range(len(graph.fragments))}

        for (i, j), var in link_vars.items():
            if model.evaluate(var) == z3.BoolVal(True):  # type: ignore
                successor_of[i] = j

        return successor_of

    def _reconstruct_tracks_from_solution(
        self, fragments: List[Track], successor_of: Dict[int, Optional[int]]
    ) -> List[Track]:
        """Reconstruct final tracks from the optimization solution."""
        # Find track starts (fragments with no predecessors)
        has_predecessor = {i: False for i in range(len(fragments))}
        for successor in successor_of.values():
            if successor is not None:
                has_predecessor[successor] = True

        starts = [i for i in range(len(fragments)) if not has_predecessor[i]]

        # Build final tracks
        final_tracks: List[Track] = []
        for track_id, start_idx in enumerate(starts, start=1):
            detections: List[Detection] = []

            # Follow the chain of fragments
            current_idx = start_idx
            while current_idx is not None:
                detections.extend(fragments[current_idx].sorted_detections)
                current_idx = successor_of[current_idx]

            # Create final track
            sorted_detections = sorted(detections, key=lambda d: d.frame_idx)
            final_tracks.append(Track(track_id=track_id, sorted_detections=sorted_detections))

        return final_tracks

    def _calculate_link_cost(self, start: Track, end: Track) -> Optional[float]:
        """Calculate the cost of linking two tracks. Returns None if they can't be linked."""
        assert end.start_frame() > start.start_frame(), 'End track must start after start track'

        if start.end_frame() < end.start_frame():
            return self._calculate_sequential_link_cost(start, end)
        else:
            return self._calculate_overlapping_link_cost(start, end)

    def _calculate_sequential_link_cost(self, start: Track, end: Track) -> Optional[float]:
        """Calculate cost for linking sequential (non-overlapping) tracks."""
        gap = end.start_frame() - start.end_frame()

        if gap > self.max_link_gap:
            return None

        # Geometric similarity (IoU)
        start_det = start.end()
        end_det = end.start()
        iou = end_det.bbox.iou(start_det.bbox)

        if iou < self.min_link_iou:
            return None

        # Appearance similarity (cosine similarity over window)
        cos = self._calculate_windowed_cosine_similarity(start, end, start_det, end_det)

        # Calculate total cost
        cost = self.w_link_iou * (1.0 - iou) + self.w_link_app * (1.0 - cos) + self.w_link_gap * gap

        return cost

    def _calculate_overlapping_link_cost(self, start: Track, end: Track) -> Optional[float]:
        """Calculate cost for linking tracks that would overlap temporally."""
        start_frames = self._get_fragment_frames(start)
        end_frames = self._get_fragment_frames(end)

        assert not start_frames.intersection(end_frames), 'Start and end tracks must not overlap'

        # Calculate gap and IoU
        min_frame = min(min(start_frames), min(end_frames))
        max_frame = max(max(start_frames), max(end_frames))
        total_frames = len(start_frames.union(end_frames))
        total_frame_duration = max_frame - min_frame
        gap = total_frame_duration - total_frames

        if gap > self.max_link_gap:
            return None

        # Find maximum IoU between any pair of detections
        max_iou = self._find_max_iou_between_tracks(start, end)

        if max_iou < self.min_link_iou:
            return None

        # Appearance similarity using average embeddings
        cos = cosine_similarity(_average_embedding(start), _average_embedding(end))

        # Calculate total cost
        cost = self.w_link_iou * (1.0 - max_iou) + self.w_link_app * (1.0 - cos) + self.w_link_gap * gap

        return cost

    def _calculate_windowed_cosine_similarity(
        self, start: Track, end: Track, start_det: Detection, end_det: Detection
    ) -> float:
        """Calculate average cosine similarity over a temporal window."""
        n_pairs = 0
        cos_sum = 0.0

        window_radius = self.link_cost_appearance_window_radius

        for i in range(-window_radius, window_radius + 1):
            d1 = start.detections_by_frame.get(start_det.frame_idx + i)
            if d1 is None:
                continue

            for j in range(-window_radius, window_radius + 1):
                d2 = end.detections_by_frame.get(end_det.frame_idx + j)
                if d2 is None:
                    continue

                cos = cosine_similarity(d1.embedding, d2.embedding)
                cos_sum += cos
                n_pairs += 1

        return cos_sum / n_pairs if n_pairs > 0 else 0.0

    def _find_max_iou_between_tracks(self, start: Track, end: Track) -> float:
        """Find the maximum IoU between any pair of detections from two tracks."""
        max_iou = 0.0

        # Compare all pairs of detections
        for start_det in start.sorted_detections:
            for end_det in end.sorted_detections:
                iou = end_det.bbox.iou(start_det.bbox)
                max_iou = max(max_iou, iou)

        # Also compare ends with all detections
        for start_det in start.sorted_detections:
            iou = end.end().bbox.iou(start_det.bbox)
            max_iou = max(max_iou, iou)

        for end_det in end.sorted_detections:
            iou = start.start().bbox.iou(end_det.bbox)
            max_iou = max(max_iou, iou)

        return max_iou
