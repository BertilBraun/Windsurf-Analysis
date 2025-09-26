from __future__ import annotations

import pulp
import logging
from typing import Dict, List, Tuple

from ..common_types import Detection, Track

from ..settings import OPTIMIZER_TIMEOUT_SECONDS


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


class ILPGraphSolver:
    """Solve an assignment problem with ILP."""

    def __init__(self, w_start: float):
        self.w_start = w_start

    def optimize_graph(self, graph: FragmentGraph) -> List[Track]:
        """Solve the fragment linking optimization problem using ILP."""
        # Create the ILP problem
        problem = pulp.LpProblem('Fragment_Linking', pulp.LpMinimize)

        # Create decision variables
        link_vars = self._create_link_variables(graph)
        start_vars = self._create_start_variables(len(graph.fragments))

        # Add constraints
        self._add_outgoing_constraints(problem, graph, link_vars)
        self._add_incoming_constraints(problem, graph, link_vars)
        self._add_start_constraints(problem, graph, link_vars, start_vars)

        # Set objective function
        self._set_objective_function(problem, graph, link_vars, start_vars)

        # Solve and return solution
        return self._solve_and_extract_solution(problem, graph, link_vars)

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
        self, problem: pulp.LpProblem, graph: FragmentGraph, link_vars: Dict[Tuple[int, int], pulp.LpVariable]
    ) -> None:
        """Add constraints ensuring each fragment has at most one outgoing link."""
        for i in range(len(graph.fragments)):
            outgoing_connections = graph.get_outgoing_connections(i)
            if outgoing_connections:
                # Sum of outgoing links <= 1
                outgoing_vars = [link_vars[(i, j)] for j in outgoing_connections]
                constraint_name = f'outgoing_{i}'
                problem += pulp.lpSum(outgoing_vars) <= 1, constraint_name

    def _add_incoming_constraints(
        self, problem: pulp.LpProblem, graph: FragmentGraph, link_vars: Dict[Tuple[int, int], pulp.LpVariable]
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
                problem += pulp.lpSum(incoming[j]) <= 1, constraint_name

    def _add_start_constraints(
        self,
        problem: pulp.LpProblem,
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
                problem += start_vars[i] + pulp.lpSum(incoming[i]) == 1, constraint_name
            else:
                # No incoming links, so must be a start
                constraint_name = f'start_forced_{i}'
                problem += start_vars[i] == 1, constraint_name

    def _set_objective_function(
        self,
        problem: pulp.LpProblem,
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
        problem += pulp.lpSum(objective_terms)

    def _solve_and_extract_solution(
        self, problem: pulp.LpProblem, graph: FragmentGraph, link_vars: Dict[Tuple[int, int], pulp.LpVariable]
    ) -> List[Track]:
        """Solve the optimization problem and extract the solution."""
        # Get solver
        solver = pulp.PULP_CBC_CMD(timeLimit=OPTIMIZER_TIMEOUT_SECONDS, msg=False)  # don't print solver output

        # Solve the problem
        problem.solve(solver)

        # Check solution status
        status = pulp.LpStatus[problem.status]

        if status == 'Optimal':
            logging.info(f'ILP solver found optimal solution with cost: {pulp.value(problem.objective)}')
        elif status == 'Feasible':
            logging.warning(f'ILP solver found feasible solution with cost: {pulp.value(problem.objective)}')
        elif status == 'Infeasible':
            raise UnsatisfiableException('Fragment linking problem is infeasible')
        elif status == 'Unbounded':
            raise RuntimeError('Fragment linking problem is unbounded')
        elif status == 'Undefined':
            raise TimeoutException('Fragment linking solver timeout or undefined status')
        else:
            raise RuntimeError(f'Unexpected solver status: {status}')

        # Extract solution
        successor_of: Dict[int, int | None] = {i: None for i in range(len(graph.fragments))}

        for (i, j), var in link_vars.items():
            if var.varValue is not None and var.varValue > 0.5:  # Binary variable is 1
                successor_of[i] = j

        return self._reconstruct_tracks_from_solution(graph.fragments, successor_of)

    def _reconstruct_tracks_from_solution(
        self, fragments: List[Track], successor_of: Dict[int, int | None]
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
