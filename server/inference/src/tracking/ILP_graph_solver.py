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
        self.fragments = fragments  # required for reconstructing tracks from solution
        self.successors: List[List[int]] = [[] for _ in range(len(fragments))]
        self.pair_costs: Dict[Tuple[int, int], float] = {}

    @property
    def N(self) -> int:
        """Get the number of fragments."""
        return len(self.fragments)

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

    def limit_outgoing_links(self, max_outgoing_links: int) -> FragmentGraph:
        """Limit the number of outgoing links to max_outgoing_links by sorting the outgoing links by cost and keeping the top max_outgoing_links."""
        simplified_graph = FragmentGraph(self.fragments)
        for i, connections in enumerate(self.successors):
            connections.sort(key=lambda x: self.pair_costs[(i, x)])  # sort by cost
            for j in connections[:max_outgoing_links]:
                simplified_graph.add_connection(i, j, self.pair_costs[(i, j)])
        return simplified_graph

    def limit_cost(self, max_cost: float) -> FragmentGraph:
        """Limit the cost of the connections to max_cost by sorting the connections by cost and keeping the top max_cost."""
        simplified_graph = FragmentGraph(self.fragments)
        for (i, j), cost in self.pair_costs.items():
            if cost <= max_cost:
                simplified_graph.add_connection(i, j, cost)
        return simplified_graph


class ILPGraphSolver:
    """Solve an assignment problem with ILP."""

    def __init__(self, max_outgoing_links: int = 10):
        self.max_outgoing_links = max_outgoing_links

    def optimize_graph(
        self,
        graph: FragmentGraph,
        start_costs: List[float],
        end_costs: List[float],
        max_cost_limit: float,
    ) -> Tuple[List[Track], float]:
        """Solve the fragment linking optimization problem using ILP."""
        # Create the ILP problem
        problem = pulp.LpProblem('Fragment_Linking', pulp.LpMinimize)

        simplified_graph = self._simplify_graph(graph, max_cost_limit)

        # Create decision variables
        link_vars = self._create_link_variables(simplified_graph)
        start_vars = self._create_start_variables(simplified_graph.N)
        end_vars = self._create_end_variables(simplified_graph.N)

        # Add constraints
        self._add_outgoing_constraints(problem, simplified_graph, link_vars, end_vars)
        self._add_incoming_constraints(problem, simplified_graph, link_vars, start_vars)

        # Set objective function
        self._set_objective_function(problem, simplified_graph, link_vars, start_vars, end_vars, start_costs, end_costs)

        # Solve and return solution
        return self._solve_and_extract_solution(problem, simplified_graph, link_vars)

    def _simplify_graph(self, graph: FragmentGraph, max_cost: float) -> FragmentGraph:
        """Simplify the graph by removing edges with cost greater than max_cost and limiting the number of outgoing links."""
        return graph.limit_outgoing_links(self.max_outgoing_links).limit_cost(max_cost)

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

    def _create_end_variables(self, num_fragments: int) -> List[pulp.LpVariable]:
        """Create binary variables for fragment ends."""
        end_vars = []
        for i in range(num_fragments):
            var_name = f'end_{i}'
            end_vars.append(pulp.LpVariable(var_name, cat='Binary'))
        return end_vars

    def _add_outgoing_constraints(
        self,
        problem: pulp.LpProblem,
        graph: FragmentGraph,
        link_vars: Dict[Tuple[int, int], pulp.LpVariable],
        end_vars: List[pulp.LpVariable],
    ) -> None:
        """Add constraints ensuring each fragment has exactly one outgoing path (link or end)."""
        for i in range(graph.N):
            outgoing_connections = graph.get_outgoing_connections(i)
            # end_i + sum(outgoing_links_from_i) == 1
            if outgoing_connections:
                outgoing_vars = [link_vars[(i, j)] for j in outgoing_connections]
                constraint_name = f'outgoing_{i}'
                problem += end_vars[i] + pulp.lpSum(outgoing_vars) == 1, constraint_name
            else:
                # No outgoing links, so must be an end
                constraint_name = f'end_forced_{i}'
                problem += end_vars[i] == 1, constraint_name

    def _add_incoming_constraints(
        self,
        problem: pulp.LpProblem,
        graph: FragmentGraph,
        link_vars: Dict[Tuple[int, int], pulp.LpVariable],
        start_vars: List[pulp.LpVariable],
    ) -> None:
        """Add constraints ensuring each fragment has exactly one incoming path (start or link)."""
        # Build incoming connections mapping
        incoming: List[List[pulp.LpVariable]] = [[] for _ in range(graph.N)]
        for (i, j), var in link_vars.items():
            incoming[j].append(var)

        # Add constraints
        for j in range(graph.N):
            # start_j + sum(incoming_links_to_j) == 1
            if incoming[j]:
                constraint_name = f'incoming_{j}'
                problem += start_vars[j] + pulp.lpSum(incoming[j]) == 1, constraint_name
            else:
                # No incoming links, so must be a start
                constraint_name = f'start_forced_{j}'
                problem += start_vars[j] == 1, constraint_name

    def _set_objective_function(
        self,
        problem: pulp.LpProblem,
        graph: FragmentGraph,
        link_vars: Dict[Tuple[int, int], pulp.LpVariable],
        start_vars: List[pulp.LpVariable],
        end_vars: List[pulp.LpVariable],
        start_costs: List[float],
        end_costs: List[float],
    ) -> None:
        """Set the objective function to minimize total cost."""
        objective_terms = []

        # Link costs
        for (i, j), var in link_vars.items():
            cost = graph.get_connection_cost(i, j)
            objective_terms.append(cost * var)

        # Start costs
        for i, var in enumerate(start_vars):
            objective_terms.append(start_costs[i] * var)

        # End costs
        for i, var in enumerate(end_vars):
            objective_terms.append(end_costs[i] * var)

        # Set objective
        problem += pulp.lpSum(objective_terms)

    def _solve_and_extract_solution(
        self, problem: pulp.LpProblem, graph: FragmentGraph, link_vars: Dict[Tuple[int, int], pulp.LpVariable]
    ) -> Tuple[List[Track], float]:
        """Solve the optimization problem and extract the solution."""
        # Get solver
        solver = pulp.PULP_CBC_CMD(timeLimit=OPTIMIZER_TIMEOUT_SECONDS, msg=False)  # don't print solver output

        # Solve the problem
        problem.solve(solver)

        # Check solution status
        status = pulp.LpStatus[problem.status]
        solution_cost = pulp.value(problem.objective)

        if status == 'Optimal':
            logging.info(f'ILP solver found optimal solution with cost: {solution_cost}')
        elif status == 'Feasible':
            logging.warning(f'ILP solver found feasible solution with cost: {solution_cost}')
        elif status == 'Infeasible':
            raise UnsatisfiableException('Fragment linking problem is infeasible')
        elif status == 'Unbounded':
            raise RuntimeError('Fragment linking problem is unbounded')
        elif status == 'Undefined':
            raise TimeoutException('Fragment linking solver timeout or undefined status')
        else:
            raise RuntimeError(f'Unexpected solver status: {status}')

        # Extract solution
        successor_of: Dict[int, int | None] = {i: None for i in range(graph.N)}

        for (i, j), var in link_vars.items():
            if var.varValue is not None and var.varValue > 0.5:  # Binary variable is 1
                successor_of[i] = j

        assert isinstance(solution_cost, float), f'Solution cost is not a float: {solution_cost}'

        return self._reconstruct_tracks_from_solution(graph.fragments, successor_of), solution_cost

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
