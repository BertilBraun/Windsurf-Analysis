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


from video_io import VideoInfo
from common_types import Detection, Track, cosine_similarity

from tracking.greedy_tracker import GreedyTracker, _average_embedding

TIMEOUT_SECONDS = 60


class TimeoutException(Exception):
    """Raised when the Z3 solver times out."""


class UnsatisfiableException(Exception):
    """Raised when the Z3 solver finds the problem unsatisfiable."""


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

    def track_detections(self, detections: list[Detection], video_properties: VideoInfo) -> list[Track]:
        logging.info(f'{"=" * 80} Running discrete optimization tracker with {len(detections)} detections {"=" * 80}')

        fragments = GreedyTracker().track_detections(detections, video_properties)
        return fragments
        logging.info(f'{"=" * 80} Running discrete optimization tracker with {len(fragments)} fragments {"=" * 80}')

        return self._optimize_fragments(fragments)

    def _optimize_fragments(self, fragments: list[Track]) -> list[Track]:
        if not fragments:
            return []
        fragments = sorted(fragments, key=lambda t: t.sorted_detections[0].frame_idx)
        N = len(fragments)

        successors: list[list[int]] = [[] for _ in range(N)]
        pair_cost: dict[tuple[int, int], float] = {}
        for i, start in enumerate(fragments):
            frames_i = {detection.frame_idx for detection in start.sorted_detections}
            for j in range(i, N):
                end = fragments[j]
                frames_j = {detection.frame_idx for detection in end.sorted_detections}
                if frames_i.intersection(frames_j):
                    # If we have any overlapping frames, we can't link them
                    continue
                cost = self._link_cost(start, end)
                if cost is None:
                    continue
                successors[i].append(j)
                pair_cost[(i, j)] = cost

        opt = z3.Optimize()
        opt.set('timeout', TIMEOUT_SECONDS * 1000)

        link_vars: dict[tuple[int, int], z3.BoolRef] = {(i, j): z3.Bool(f'link_{i}_{j}') for (i, j) in pair_cost}

        for i in range(N):
            out_links = [link_vars[(i, j)] for j in successors[i]]
            if out_links:
                opt.add(z3.PbLe([(v, 1) for v in out_links], 1))

        incoming: list[list[z3.BoolRef]] = [[] for _ in range(N)]
        for (i, j), v in link_vars.items():
            incoming[j].append(v)

        for j in range(N):
            if incoming[j]:
                opt.add(z3.PbLe([(v, 1) for v in incoming[j]], 1))

        start_vars: list[z3.BoolRef] = [z3.Bool(f'start_{i}') for i in range(N)]

        for i in range(N):
            if incoming[i]:
                opt.add(start_vars[i] == z3.And([z3.Not(v) for v in incoming[i]]))
            else:
                opt.add(start_vars[i])

        link_cost_terms = [z3.If(v, z3.RealVal(pair_cost[(i, j)]), z3.RealVal(0.0)) for (i, j), v in link_vars.items()]
        start_cost_terms = [z3.If(sv, z3.RealVal(self.w_start), z3.RealVal(0.0)) for sv in start_vars]
        total_cost = (
            z3.Sum(link_cost_terms + start_cost_terms) if (link_cost_terms or start_cost_terms) else z3.RealVal(0.0)
        )
        opt.minimize(total_cost)

        res = opt.check()
        if res != z3.sat:
            if res == z3.unknown:
                raise TimeoutException('Fragment linking solver timeout')
            if res == z3.unsat:
                raise UnsatisfiableException('Fragment linking UNSAT')
            raise RuntimeError(f'Unexpected solver status {res}')

        model = opt.model()
        successor_of: dict[int, int | None] = {i: None for i in range(N)}
        has_predecessor: dict[int, bool] = {i: False for i in range(N)}
        for (i, j), v in link_vars.items():
            if model.evaluate(v) == z3.BoolVal(True):  # type: ignore
                successor_of[i] = j
                has_predecessor[j] = True
        starts = [i for i in range(N) if not has_predecessor[i]]

        final_tracks: list[Track] = []
        for track_id, start_idx in enumerate(starts, start=1):
            detections: list[Detection] = []

            cur = start_idx
            while cur is not None:
                detections.extend(fragments[cur].sorted_detections)
                cur = successor_of[cur]

            sorted_detections = list(sorted(detections, key=lambda d: d.frame_idx))
            final_tracks.append(Track(track_id=track_id, sorted_detections=sorted_detections))

        return final_tracks

    def _link_cost(self, start: Track, end: Track) -> float | None:
        """Calculates link cost between two tracks [0-1]. Returns None if the tracks can't be connected."""
        assert end.start_frame() > start.start_frame(), 'End track must start after start track'
        if start.end_frame() < end.start_frame():
            # We merge end behind start

            gap = end.start_frame() - start.end_frame()

            start_det = start.end()
            end_det = end.start()
            iou = end_det.bbox.iou(start_det.bbox)
            # average cosine similarity over similarity for link_appearance_window_radius frames

            n_pairs = 0
            cos_sum = 0.0
            for i in range(-self.link_cost_appearance_window_radius, self.link_cost_appearance_window_radius + 1):
                d1 = start.detections_by_frame.get(start_det.frame_idx + i)
                if d1 is None:
                    continue
                for j in range(-self.link_cost_appearance_window_radius, self.link_cost_appearance_window_radius + 1):
                    d2 = end.detections_by_frame.get(end_det.frame_idx + j)
                    if d2 is None:
                        continue
                    cos = cosine_similarity(d1.embedding, d2.embedding)
                    cos_sum += cos
                    n_pairs += 1
            cos = cos_sum / n_pairs if n_pairs > 0 else 0.0

        else:
            # We merge end into the middle start
            start_frames = {d.frame_idx for d in start.sorted_detections}
            end_frames = {d.frame_idx for d in end.sorted_detections}

            assert not start_frames.intersection(end_frames), 'Start and end tracks must not overlap'

            start_frame = min(min(start_frames), min(end_frames))
            end_frame = max(max(start_frames), max(end_frames))

            total_frames = len(start_frames.union(end_frames))
            total_frame_duration = end_frame - start_frame

            gap = total_frame_duration - total_frames

            i, j = 0, 0
            max_iou_between_successive_detections = 0.0
            while i < len(start.sorted_detections) and j < len(end.sorted_detections):
                start_det = start.sorted_detections[i]
                end_det = end.sorted_detections[j]
                iou = end_det.bbox.iou(start_det.bbox)
                max_iou_between_successive_detections = max(max_iou_between_successive_detections, iou)
                if start_det.frame_idx < end_det.frame_idx:
                    i += 1
                else:
                    j += 1
            for i in range(len(start.sorted_detections)):
                iou_end = end.end().bbox.iou(start.sorted_detections[i].bbox)
                max_iou_between_successive_detections = max(max_iou_between_successive_detections, iou_end)
            for j in range(len(end.sorted_detections)):
                iou_start = start.start().bbox.iou(end.sorted_detections[j].bbox)
                max_iou_between_successive_detections = max(max_iou_between_successive_detections, iou_start)

            iou = max_iou_between_successive_detections

            cos = cosine_similarity(_average_embedding(start), _average_embedding(end))

        if gap > self.max_link_gap:
            return None

        if iou < self.min_link_iou:
            return None

        # cos = cosine_similarity(end_det.feat, start_det.feat)
        # if cos < self.min_link_cos:
        #     assert False, "Let the solver handle this for now."
        #     return None
        cost = self.w_link_iou * (1.0 - iou) + self.w_link_app * (1.0 - cos) + self.w_link_gap * gap
        return cost
