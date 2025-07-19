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

from enum import Enum
from collections import defaultdict


from video_io import VideoInfo
from common_types import Detection, FrameIndex, Track, TrackId, cosine_similarity

MAX_TRACKS = 6
TIMEOUT_S = 30

# Cost weights (tune as needed)
W_LOCAL_IOU = 1.0
W_EMB_WINDOW = 1.0   # weight for sliding-window embedding cohesion

# Sliding window parameters
EMB_WINDOW_FRAMES = 5 * 30      # W: max |frame_j - frame_i| to consider
EMB_NEIGHBORS_FWD = 5 * 30      # K_next
EMB_NEIGHBORS_BACK = 5 * 30     # K_prev
MIN_COS_FOR_WINDOW = -1.0   # optionally gate pairs by similarity (keep all)
WINDOW_STRIDE = 10



def count_max_simultaneous_detections(
    detections: list[Detection],
) -> int:
    n_detections = defaultdict(int)
    for det in detections:
        n_detections[det.frame_idx] += 1
    return max(n_detections.values(), default=0)


class TimeoutException(Exception):
    """Raised when the Z3 solver times out."""


class UnsatisfiableException(Exception):
    """Raised when the Z3 solver finds the problem unsatisfiable."""



class _ComparisonResult(Enum):
    MATCH = 'match'
    MAY_MATCH = 'may_match'
    NO_MATCH = 'no_match'


class DiscreteOptimizationTracker:
    """Track objects by solving an assignment problem with *Z3*."""

    def __init__(
        self,
        greedy_min_iou: float = 0.8,
        greedy_min_cosine_similarity: float = 0.5,
        greedy_max_frame_distance: int = 5,
        greedy_min_iou_matches_single_track: float = 0.2,
        use_fragment_linking: bool = True,
        max_link_gap: int = 30,
        min_link_iou: float = 0.0,
        min_link_cos: float = -1.0,
        w_link_iou: float = 1.0,
        w_link_app: float = 1.0,
        w_link_gap: float = 0.005,
        w_start: float = 10.0, # <-- should be scaled according to number of estimated starts / tracks and number links required
    ):
        self.greedy_min_iou = greedy_min_iou
        self.greedy_min_cosine_similarity = greedy_min_cosine_similarity
        self.greedy_max_frame_distance = greedy_max_frame_distance
        self.greedy_min_iou_matches_single_track = greedy_min_iou_matches_single_track
        # Fragment linking config
        self.use_fragment_linking = use_fragment_linking
        self.max_link_gap = max_link_gap
        self.min_link_iou = min_link_iou
        self.min_link_cos = min_link_cos
        self.w_link_iou = w_link_iou
        self.w_link_app = w_link_app
        self.w_link_gap = w_link_gap
        self.w_start = w_start
        self.min_iou_matches_single_track = greedy_min_iou_matches_single_track

    def _compare_detection_to_track(self, track: Track, detection: Detection) -> _ComparisonResult:
        iou = track.end().bbox.iou(detection.bbox)

        if iou < self.min_iou_matches_single_track:
            return _ComparisonResult.NO_MATCH

        # TODO cosine similarity to more than just one frame?
        n = len(track.sorted_detections)
        average_sim = sum(cosine_similarity(d.feat, detection.feat) for d in track.sorted_detections) / n

        if iou >= self.greedy_min_iou and average_sim >= self.greedy_min_cosine_similarity:
            return _ComparisonResult.MATCH

        return _ComparisonResult.MAY_MATCH

    def _preprocess_detections(self, detections_by_frame: dict[int, list[Detection]]) -> list[Track]:
        """Greedily stiches detections onto tracks as long as both IOU and cosine similarity are high."""

        # We match greedily if:
        # the bounding box of a detection overlaps only with a single active track
        # or both iou and cosine similarity are high enough to continue the track.
        #
        # We only match against active tracks.
        # Tracks become stale if they:
        # - are too old
        # - have been considered for a match but ot chosen
        # - have been matched by multiple detections in the same frame

        # Sort detections by frame index to process them in order.
        sorted_frame_indices = sorted(detections_by_frame.keys())

        next_track_id = 1

        # these tracks have been detected and can be matched to further detections
        active_tracks: list[Track] = []
        # these tracks have been detected but can't match further detections
        stale_tracks: list[Track] = []
        stale_track_ids: set[TrackId] = set()

        for frame_idx in sorted_frame_indices:
            matches_this_frame = []
            for detection in detections_by_frame[frame_idx]:
                clean_matches: list[Track] = []
                mby_matches: list[Track] = []

                for track in active_tracks:
                    comparison_result = self._compare_detection_to_track(track, detection)
                    if comparison_result == _ComparisonResult.MATCH:
                        # Track matches the detection, continue it
                        clean_matches.append(track)
                    elif comparison_result == _ComparisonResult.MAY_MATCH:
                        mby_matches.append(track)

                if len(clean_matches) == 1:
                    matches_this_frame.append(
                        (clean_matches[0], detection)
                    )
                    # track = clean_matches[0]
                    # track.sorted_detections.append(detection)
                elif len(clean_matches) == 0 and len(mby_matches) == 1:
                    # track = mby_matches[0]
                    # track.sorted_detections.append(detection)
                    matches_this_frame.append(
                        (mby_matches[0], detection)
                    )
                else:
                    # no clear match found, create a new track for this detection
                    new_track = Track(track_id=next_track_id, sorted_detections=[])
                    matches_this_frame.append(
                        (new_track, detection)
                    )
                    # active_tracks.append(new_track)
                    next_track_id += 1

                    # all clean and mby matches are stale because we couldn't cleanly match them
                    for track in clean_matches + mby_matches:
                        if track.track_id not in stale_track_ids:
                            stale_track_ids.add(track.track_id)
                            stale_tracks.append(track)
            # now we construct tracks for any match that is not a duplicate. Tracks that are matched by multiple detections will become stale (the detections will be assigned to the new track).
            detections_per_track = defaultdict(list)
            tracks_per_track_id = {}
            for track, detection in matches_this_frame:
                detections_per_track[track.track_id].append(detection)
                tracks_per_track_id[track.track_id] = track
            for track_id, dets in detections_per_track.items():
                track = tracks_per_track_id[track_id]
                if len(dets) > 1:
                    stale_track_ids.add(track_id)
                    stale_tracks.append(track)
                    for detection in dets:
                        active_tracks.append(Track(track_id=next_track_id, sorted_detections=[detection]))
                        next_track_id += 1
                else:
                    assert len(dets) == 1
                    track.sorted_detections.append(dets[0])
                    # JANK
                    if track not in active_tracks:
                        active_tracks.append(track)

            # Too old tracks are stale
            for track in active_tracks:
                if track.end().frame_idx + self.greedy_max_frame_distance < frame_idx:
                    if track.track_id not in stale_track_ids:
                        stale_track_ids.add(track.track_id)
                        stale_tracks.append(track)
            # update stale / active tracks once per frame
            active_tracks = [track for track in active_tracks if track not in stale_tracks]

        return stale_tracks + active_tracks

    def _link_cost(
            self,
            start: Track,
            end: Track,
    ) -> float | None:
        """Calculates link cost between two tracks [0-1]. Returns None if the tracks can't be connected."""
        assert start.end_frame() < end.start_frame(), "Tracks must not overlap."
        start_det = start.end()
        end_det = end.start()
        gap = end.start_frame() - start.end_frame()
        if gap > self.max_link_gap:
            return None
        iou = end_det.bbox.iou(start_det.bbox)
        if iou < self.min_link_iou:
            assert False, "Let the solver handlet his for now."
            return None
        cos = cosine_similarity(end_det.feat, start_det.feat)
        if cos < self.min_link_cos:
            assert False, "Let the solver handlet his for now."
            return None
        cost = (
            self.w_link_iou * (1.0 - iou)
            + self.w_link_app * (1.0 - cos)
            + self.w_link_gap * gap
        )
        return cost

    # ------------------------------------------------------------------
    # Fragment linking optimisation (new mode)
    # ------------------------------------------------------------------
    def _optimize_fragments(
        self,
        fragments: List[Track],
        n_tracks: int | None,
    ) -> List[Track]:
        """If n_tracks is given we require n_tracks to be generated."""
        if not fragments:
            return []
        fragments = sorted(fragments, key=lambda t: t.sorted_detections[0].frame_idx)
        F = len(fragments)

        successors: List[List[int]] = [[] for _ in range(F)]
        pair_cost: dict[tuple[int,int], float] = {}
        for i, start in enumerate(fragments):
            for j in range(i, F):
                end = fragments[j]
                if end.start_frame() <= start.end_frame():
                    continue
                cost = self._link_cost(start, end)
                if cost is None:
                    continue
                successors[i].append(j)
                pair_cost[(i,j)] = cost

        opt = z3.Optimize()
        opt.set("timeout", TIMEOUT_S * 1000)
        link_vars: dict[tuple[int,int], z3.BoolRef] = {(i,j): z3.Bool(f"link_{i}_{j}") for (i,j) in pair_cost}
        start_vars: List[z3.BoolRef] = [z3.Bool(f"start_{i}") for i in range(F)]

        for i in range(F):
            out_links = [link_vars[(i,j)] for j in successors[i]]
            if out_links:
                opt.add(z3.PbLe([(v,1) for v in out_links], 1))
        incoming: List[List[z3.BoolRef]] = [[] for _ in range(F)]
        for (i,j), v in link_vars.items():
            incoming[j].append(v)
        for j in range(F):
            inc = incoming[j]
            if inc:
                opt.add(z3.PbLe([(v,1) for v in inc], 1))
        for i in range(F):
            inc = incoming[i]
            if not inc:
                opt.add(start_vars[i])
            else:
                opt.add(start_vars[i] == z3.And([z3.Not(v) for v in inc]))
        if n_tracks is not None:
            opt.add(z3.PbLe([(sv,1) for sv in start_vars], n_tracks))

        link_cost_terms = [z3.If(v, z3.RealVal(str(pair_cost[(i,j)])), z3.RealVal("0")) for (i,j), v in link_vars.items()]
        start_cost_terms = [z3.If(sv, z3.RealVal(str(self.w_start)), z3.RealVal("0")) for sv in start_vars]
        total_cost = z3.Sum(link_cost_terms + start_cost_terms) if (link_cost_terms or start_cost_terms) else z3.RealVal("0")
        opt.minimize(total_cost)

        res = opt.check()
        if res != z3.sat:
            if res == z3.unknown:
                raise TimeoutException("Fragment linking solver timeout")
            if res == z3.unsat:
                raise UnsatisfiableException("Fragment linking UNSAT")
            raise RuntimeError(f"Unexpected solver status {res}")

        model = opt.model()
        succ_of = {i: None for i in range(F)}
        has_pred = {i: False for i in range(F)}
        for (i,j), v in link_vars.items():
            if model.evaluate(v) == z3.BoolVal(True):
                succ_of[i] = j
                has_pred[j] = True
        starts = [i for i in range(F) if not has_pred[i]]
        final_tracks: List[Track] = []
        next_track_id = 0
        for s in starts:
            chain_indices = []
            cur = s
            while cur is not None:
                chain_indices.append(cur)
                cur = succ_of[cur]
            dets: List[Detection] = []
            for idx in chain_indices:
                dets.extend(fragments[idx].sorted_detections)
            final_tracks.append(Track(track_id=next_track_id, sorted_detections=dets))
            next_track_id += 1
        return final_tracks

    # ------------------------------------------------------------------
    # Core optimisation (detection-level)  (legacy mode)
    # ------------------------------------------------------------------

    def _track_detections_inner(
        self,
        detections: list[Detection],
        n_tracks: int,
        must_link_groups: Iterable[list[Detection]] | None = None,
    ) -> List[Track]:
        if not detections:
            return []
        detections = sorted(detections, key=lambda d: d.frame_idx)
        num_det = len(detections)
        det_index = {id(det): idx for idx, det in enumerate(detections)}

        frame_index_to_assignment_indices: dict[FrameIndex, list[int]] = defaultdict(list)
        for idx, det in enumerate(detections):
            frame_to_indices[det.frame_idx].append(idx)
        frames_sorted = sorted(frame_to_indices.keys())

        opt = z3.Optimize()
        opt.set("timeout", TIMEOUT_S * 1000)

        assign: list[z3.IntNumRef] = [z3.Int(f"track_{idx}") for idx in range(num_det)]
        for var in assign:
            opt.add(z3.And(var >= 0, var < n_tracks))

        # Frame-level AllDifferent
        for indices in frame_to_indices.values():
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    opt.add(assign[indices[a]] != assign[indices[b]])

        # Must-link constraints
        if must_link_groups:
            seen = set()
            for group in must_link_groups:
                idxs = [det_index.get(id(d)) for d in group]
                idxs = [i for i in idxs if i is not None]
                if len(idxs) < 2:
                    continue
                for i in idxs:
                    if i in seen:
                        raise ValueError("Detection appears in multiple must-link groups; groups must be disjoint.")
                    seen.add(i)
                base = assign[idxs[0]]
                for i in idxs[1:]:
                    opt.add(assign[i] == base)

        # Local IoU continuity cost (adjacent frames only)
        local_terms: List[z3.ArithRef] = []
        for f_idx in range(len(frames_sorted) - 1):
            f_curr, f_next = frames_sorted[f_idx], frames_sorted[f_idx + 1]
            for i in frame_to_indices[f_curr]:
                for j in frame_to_indices[f_next]:
                    iou = detections[i].bbox.iou(detections[j].bbox)
                    iou = max(0.0, min(1.0, iou))
                    cost_val = 1.0 - iou
                    if cost_val == 0:
                        continue
                    local_terms.append(
                        z3.If(assign[i] == assign[j], z3.RealVal(str(W_LOCAL_IOU * cost_val)), z3.RealVal("0"))
                    )

        # Sliding-window embedding cohesion
        # Build neighbor lists (forward only edges to avoid double counting)
        emb_terms: List[z3.ArithRef] = []
        # Precompute list of indices sorted by frame (already sorted) for rapid access
        indices_by_frame = [idx for idx in range(num_det)]  # detections already sorted

        # For each detection i, gather backward and forward neighbor candidates within frame window
        # but only create terms for forward neighbors (j>i). Use per-detection normalisation.
        for pos, i in enumerate(indices_by_frame):
            frame_i = detections[i].frame_idx
            # Collect candidate backward and forward indices inside frame window
            # We'll use them only to compute normalisation count for i (N_i) but only add forward edges.
            # Backward neighbors (indices < pos)
            back = []
            for k in range(1, EMB_NEIGHBORS_BACK + 1, WINDOW_STRIDE):
                if pos - k < 0:
                    break
                j_idx = indices_by_frame[pos - k]
                if frame_i - detections[j_idx].frame_idx > EMB_WINDOW_FRAMES:
                    break  # frames further away (since sorted, earlier ones will be even farther)
                back.append(j_idx)
            # Forward neighbors (indices > pos)
            fwd = []
            for k in range(1, EMB_NEIGHBORS_FWD + 1, WINDOW_STRIDE):
                if pos + k >= len(indices_by_frame):
                    break
                j_idx = indices_by_frame[pos + k]
                if detections[j_idx].frame_idx - frame_i > EMB_WINDOW_FRAMES:
                    break
                fwd.append(j_idx)

            # Normalisation count for i = total neighbors (back + forward) actually considered
            norm_count = len(back) + len(fwd)
            if norm_count == 0:
                continue
            norm_factor = 1.0 / norm_count

            # Add terms only for forward edges (i -> j in fwd)
            for j_idx in fwd:
                cos_ij = cosine_similarity(detections[i].feat, detections[j_idx].feat)
                if cos_ij < MIN_COS_FOR_WINDOW:
                    continue
                cos_ij = max(-1.0, min(1.0, cos_ij))
                emb_cost = 1.0 - cos_ij  # Penalise dissimilar embeddings on same track
                if emb_cost == 0:
                    continue
                emb_terms.append(
                    z3.If(
                        assign[i] == assign[j_idx],
                        z3.RealVal(str(W_EMB_WINDOW * emb_cost * norm_factor)),
                        z3.RealVal("0"),
                    )
                )

        total_cost = z3.Sum(local_terms + emb_terms) if (local_terms or emb_terms) else z3.RealVal("0")
        opt.minimize(total_cost)

        result = opt.check()
        if result != z3.sat:
            if result == z3.unknown:
                logging.debug("Z3 solver timed out")
                raise TimeoutException("Z3 solver timed out")
            if result == z3.unsat:
                logging.debug("Z3 solver found the problem unsatisfiable")
                raise UnsatisfiableException("Z3 solver found the problem unsatisfiable")
            raise RuntimeError(f"Unexpected Z3 solver status: {result}")

        model = opt.model()
        assign_vals = [model.evaluate(v).as_long() for v in assign]
        track_to_dets: Dict[int, List[Detection]] = defaultdict(list)
        for det, tid in zip(detections, assign_vals):
            track_to_dets[tid].append(det)
        tracks: List[Track] = []
        for tid, dets in track_to_dets.items():
            dets_sorted = sorted(dets, key=lambda d: d.frame_idx)
            tracks.append(Track(track_id=tid, sorted_detections=dets_sorted))
        return tracks

    def track_detections(
        self,
        detections: list[Detection],
        video_properties: VideoInfo | None = None,
    ) -> list[Track]:
        logging.info(
            f"{'=' * 80} Running discrete optimization tracker with {len(detections)} detections {'=' * 80}")
        detections_by_frame: dict[FrameIndex, list[Detection]] = defaultdict(list)
        for det in detections:
            detections_by_frame[det.frame_idx].append(det)
        max_detections = count_max_simultaneous_detections(detections)
        logging.info(f"Max simultaneous detections: {max_detections}")
        fragments = self._preprocess_detections(detections_by_frame)
        return fragments
        return self._optimize_fragments(fragments, None)



        # must_link_groups = [t.sorted_detections for t in fragments if len(t.sorted_detections) > 1]
        # logging.info(
        #     f"Pre-processing produced {len(fragments)} fragments; {len(must_link_groups)} must-link groups (len>1)."
        # )
        # n_tracks = max_detections
        # logging.debug(f"Invoking solver with n_tracks={n_tracks}")
        # solution_tracks = self._track_detections_inner(
        #     detections, n_tracks=n_tracks, must_link_groups=must_link_groups
        # )
        # logging.info(f"Solver produced {len(solution_tracks)} tracks")
        # return solution_tracks
