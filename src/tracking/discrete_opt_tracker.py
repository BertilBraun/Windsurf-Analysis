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

import pulp
import logging
from typing import Dict, List, Tuple, Optional

from similarity_helpers import cosine_similarity, mean_embedding_cosine_similarity
from video_io import VideoInfo
from common_types import Detection, Track

from settings import (
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
    OPTIMIZER_TIMEOUT_SECONDS,
)


class TimeoutException(Exception):
    """Raised when the ILP solver times out."""


class UnsatisfiableException(Exception):
    """Raised when the ILP solver finds the problem unsatisfiable."""


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


class DiscreteILPTracker:
    """Track objects by solving an assignment problem with ILP."""

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
        self.video_fps = -1  # set in track()

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

    def track(self, tracks: List[Track], video_properties: VideoInfo) -> List[Track]:
        """Main entry point for tracking detections."""
        logging.info(f'{"=" * 30} Running ILP discrete optimization tracker with {len(tracks)} tracks {"=" * 30}')

        if not tracks:
            logging.warning('No tracks available for processing')
            return []

        self.video_fps = video_properties.fps

        return self._optimize_fragments(tracks)

    def _optimize_fragments(self, fragments: List[Track]) -> List[Track]:
        """Optimize fragment connections using ILP solver."""
        # Sort fragments by start frame
        fragments = sorted(fragments, key=lambda t: t.start_frame())

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
            for j in range(i, N):
                end_fragment = fragments[j]

                # Skip if fragments have overlapping frames
                gap = end_fragment.start_frame() - start_fragment.end_frame()
                if gap <= 0 or gap > self.video_fps * MAX_OVERLAP_LENGTH_SECONDS:
                    continue

                # Calculate connection cost
                cost = self._calculate_link_cost(start_fragment, end_fragment)
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

    def _get_adaptive_parameters(self, track_length_frames: int) -> AdaptiveParameters:
        """Get adaptive parameters based on track length with linear interpolation."""
        short_track_threshold_frames = self.short_track_threshold_seconds * self.video_fps
        long_track_threshold_frames = self.long_track_threshold_seconds * self.video_fps

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

        return DiscreteILPTracker.AdaptiveParameters(
            min_link_iou, min_link_cos, w_link_iou, w_link_app, w_link_gap, window_radius
        )

    def _solve_optimization_problem(self, graph: FragmentGraph) -> Dict[int, Optional[int]]:
        """Solve the fragment linking optimization problem using ILP."""
        # Create the ILP problem
        prob = pulp.LpProblem('Fragment_Linking', pulp.LpMinimize)

        # Create decision variables
        link_vars = self._create_link_variables(graph)
        start_vars = self._create_start_variables(len(graph.fragments))

        # Add constraints
        self._add_outgoing_constraints(prob, graph, link_vars)
        self._add_incoming_constraints(prob, graph, link_vars)
        self._add_start_constraints(prob, graph, link_vars, start_vars)

        # Set objective function
        self._set_objective_function(prob, graph, link_vars, start_vars)

        # Solve and return solution
        return self._solve_and_extract_solution(prob, graph, link_vars)

    def _create_link_variables(self, graph: FragmentGraph) -> Dict[Tuple[int, int], pulp.LpVariable]:
        """Create binary variables for fragment links."""
        link_vars = {}
        for i, j in graph.get_all_connections():
            var_name = f'link_{i}_{j}'
            link_vars[(i, j)] = pulp.LpVariable(var_name, cat='Binary')
        return link_vars

    def _create_start_variables(self, num_fragments: int) -> List[pulp.LpVariable]:
        """Create binary variables for fragment starts."""
        start_vars = []
        for i in range(num_fragments):
            var_name = f'start_{i}'
            start_vars.append(pulp.LpVariable(var_name, cat='Binary'))
        return start_vars

    def _add_outgoing_constraints(
        self, prob: pulp.LpProblem, graph: FragmentGraph, link_vars: Dict[Tuple[int, int], pulp.LpVariable]
    ) -> None:
        """Add constraints ensuring each fragment has at most one outgoing link."""
        for i in range(len(graph.fragments)):
            outgoing_connections = graph.get_outgoing_connections(i)
            if outgoing_connections:
                # Sum of outgoing links <= 1
                outgoing_vars = [link_vars[(i, j)] for j in outgoing_connections]
                constraint_name = f'outgoing_{i}'
                prob += pulp.lpSum(outgoing_vars) <= 1, constraint_name

    def _add_incoming_constraints(
        self, prob: pulp.LpProblem, graph: FragmentGraph, link_vars: Dict[Tuple[int, int], pulp.LpVariable]
    ) -> None:
        """Add constraints ensuring each fragment has at most one incoming link."""
        # Build incoming connections mapping
        incoming: List[List[pulp.LpVariable]] = [[] for _ in range(len(graph.fragments))]
        for (i, j), var in link_vars.items():
            incoming[j].append(var)

        # Add constraints
        for j in range(len(graph.fragments)):
            if incoming[j]:
                constraint_name = f'incoming_{j}'
                prob += pulp.lpSum(incoming[j]) <= 1, constraint_name

    def _add_start_constraints(
        self,
        prob: pulp.LpProblem,
        graph: FragmentGraph,
        link_vars: Dict[Tuple[int, int], pulp.LpVariable],
        start_vars: List[pulp.LpVariable],
    ) -> None:
        """Add constraints defining when a fragment is a start of a track."""
        # Build incoming connections mapping
        incoming: List[List[pulp.LpVariable]] = [[] for _ in range(len(graph.fragments))]
        for (i, j), var in link_vars.items():
            incoming[j].append(var)

        # Add start constraints
        for i in range(len(graph.fragments)):
            if incoming[i]:
                # start_i = 1 - sum(incoming_links_to_i)
                # This means: start_i + sum(incoming_links_to_i) = 1
                constraint_name = f'start_{i}'
                prob += start_vars[i] + pulp.lpSum(incoming[i]) == 1, constraint_name
            else:
                # No incoming links, so must be a start
                constraint_name = f'start_forced_{i}'
                prob += start_vars[i] == 1, constraint_name

    def _set_objective_function(
        self,
        prob: pulp.LpProblem,
        graph: FragmentGraph,
        link_vars: Dict[Tuple[int, int], pulp.LpVariable],
        start_vars: List[pulp.LpVariable],
    ) -> None:
        """Set the objective function to minimize total cost."""
        objective_terms = []

        # Link costs
        for (i, j), var in link_vars.items():
            cost = graph.get_connection_cost(i, j)
            objective_terms.append(cost * var)

        # Start costs
        for var in start_vars:
            objective_terms.append(self.w_start * var)

        # Set objective
        if objective_terms:
            prob += pulp.lpSum(objective_terms)
        else:
            # Empty objective if no terms
            prob += 0

    def _solve_and_extract_solution(
        self, prob: pulp.LpProblem, graph: FragmentGraph, link_vars: Dict[Tuple[int, int], pulp.LpVariable]
    ) -> Dict[int, Optional[int]]:
        """Solve the optimization problem and extract the solution."""
        # Get solver
        solver = pulp.PULP_CBC_CMD(timeLimit=OPTIMIZER_TIMEOUT_SECONDS, msg=False)  # don't print solver output

        # Solve the problem
        prob.solve(solver)

        # Check solution status
        status = pulp.LpStatus[prob.status]

        if status == 'Optimal':
            logging.info(f'ILP solver found optimal solution with cost: {pulp.value(prob.objective)}')
        elif status == 'Feasible':
            logging.warning(f'ILP solver found feasible solution with cost: {pulp.value(prob.objective)}')
        elif status == 'Infeasible':
            raise UnsatisfiableException('Fragment linking problem is infeasible')
        elif status == 'Unbounded':
            raise RuntimeError('Fragment linking problem is unbounded')
        elif status == 'Undefined':
            raise TimeoutException('Fragment linking solver timeout or undefined status')
        else:
            raise RuntimeError(f'Unexpected solver status: {status}')

        # Extract solution
        successor_of: Dict[int, Optional[int]] = {i: None for i in range(len(graph.fragments))}

        for (i, j), var in link_vars.items():
            if var.varValue is not None and var.varValue > 0.5:  # Binary variable is 1
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
        gap = end.start_frame() - start.end_frame()

        assert gap >= 0, 'Gap must be non-negative'
        assert gap <= self.video_fps * MAX_OVERLAP_LENGTH_SECONDS, 'Gap must be less than max overlap length'

        # Get adaptive parameters based on both tracks
        # Use the shorter DURATION (in frames) of both tracks for parameter selection,
        # not the number of detections. This better reflects temporal extent.
        start_duration_frames = start.end_frame() - start.start_frame() + 1
        end_duration_frames = end.end_frame() - end.start_frame() + 1
        min_duration_frames = min(start_duration_frames, end_duration_frames)

        adaptive_params = self._get_adaptive_parameters(min_duration_frames)

        # Geometric similarity (IoU)
        iou = end.start().bbox.iou(start.end().bbox)

        if iou < adaptive_params.min_link_iou:
            return None

        # Appearance similarity
        if adaptive_params.window_radius >= min_duration_frames:  # Use mean appearance for long tracks
            cos = mean_embedding_cosine_similarity(start, end)
        else:
            cos = self._calculate_windowed_cosine_similarity(start, end, adaptive_params.window_radius)

        if cos < adaptive_params.min_link_cos:
            return None

        gap_percentage = gap / (self.video_fps * MAX_OVERLAP_LENGTH_SECONDS)

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
            d1 = start.detections_by_frame.get(start.end_frame() + i)
            if d1 is None:
                continue

            for j in range(-window_radius, window_radius + 1):
                d2 = end.detections_by_frame.get(end.start_frame() + j)
                if d2 is None:
                    continue

                cos = cosine_similarity(d1.embedding, d2.embedding)
                cos_sum += cos
                n_pairs += 1

        return cos_sum / n_pairs if n_pairs > 0 else 0.0
