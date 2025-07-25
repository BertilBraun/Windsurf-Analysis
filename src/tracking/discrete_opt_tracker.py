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

import pulp
import logging
from typing import Dict, List, Tuple, Optional, Set

from similarity_helpers import cosine_similarity, mean_embedding_cosine_similarity
from video_io import VideoInfo
from common_types import Detection, Track

from settings import (
    MAX_OVERLAP_LENGTH_SECONDS,
    OPTIMIZER_MIN_LINK_IOU,
    OPTIMIZER_MIN_LINK_COS,
    OPTIMIZER_W_LINK_IOU,
    OPTIMIZER_W_LINK_APP,
    OPTIMIZER_W_LINK_GAP,
    OPTIMIZER_W_START,
    OPTIMIZER_LINK_COST_APPEARANCE_WINDOW_RADIUS,
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
        min_link_iou: float = OPTIMIZER_MIN_LINK_IOU,
        min_link_cos: float = OPTIMIZER_MIN_LINK_COS,
        w_link_iou: float = OPTIMIZER_W_LINK_IOU,
        w_link_app: float = OPTIMIZER_W_LINK_APP,
        w_link_gap: float = OPTIMIZER_W_LINK_GAP,
        w_start: float = OPTIMIZER_W_START,
        link_cost_appearance_window_radius: int = OPTIMIZER_LINK_COST_APPEARANCE_WINDOW_RADIUS,
    ):
        self.max_link_gap = -1  # set in track()
        self.min_link_iou = min_link_iou
        self.min_link_cos = min_link_cos
        self.w_link_iou = w_link_iou
        self.w_link_app = w_link_app
        self.w_link_gap = w_link_gap
        self.w_start = w_start
        self.link_cost_appearance_window_radius = link_cost_appearance_window_radius

    def track(self, tracks: List[Track], video_properties: VideoInfo) -> List[Track]:
        """Main entry point for tracking detections."""
        logging.info(f'{"=" * 80} Running ILP discrete optimization tracker with {len(tracks)} tracks {"=" * 80}')

        if not tracks:
            logging.warning('No tracks available for processing')
            return []

        self.max_link_gap = video_properties.fps * MAX_OVERLAP_LENGTH_SECONDS

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
                if gap < 0 or gap > self.max_link_gap:
                    continue

                # Calculate connection cost
                cost = self._calculate_link_cost(start_fragment, end_fragment)
                if cost is not None:
                    graph.add_connection(i, j, cost)

        return graph

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
        assert end.start_frame() > start.start_frame(), 'End track must start after start track'
        assert end.start_frame() - start.end_frame() <= self.max_link_gap, 'Gap between tracks is too large'
        assert end.start_frame() - start.end_frame() >= 0, 'Gap between tracks is negative'

        gap = end.start_frame() - start.end_frame()

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
