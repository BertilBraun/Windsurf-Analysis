from __future__ import annotations

"""Discrete‑optimization based multi‑object tracker using Z3.

This implementation follows the design sketched in the conversation and now
adds *must‑link* constraints produced by a greedy pre‑processing pass:
    • A *decision variable* per detection selects which track the detection
      belongs to (domain: 0‥n_tracks‑1).
    • *Frame‑level exclusivity*: no two detections that appear in the same
      frame may share the same track id. This is encoded with pair‑wise
      inequality constraints.
    • *Greedy pre‑processing*: we build short, trivially obvious track
      fragments (high IoU + appearance similarity or unique overlap). Each
      such fragment yields a **must‑link group**: all detections inside the
      fragment must share the same final track id in the optimisation phase.
    • *Cost function*: when two detections in consecutive frames belong to the
      same track, the model accrues a cost equal to

            (1 – IoU(bbox_prev, bbox_curr))
          + (1 – cosine_similarity(feat_prev, feat_curr))

      Lower cost ⇒ nicer continuity. Only **adjacent** frames are considered
      to keep the formulation compact.  Supporting arbitrary frame gaps would
      need helper variables (see earlier iteration) and can be reinstated.

    • The total cost is minimised with a Z3 `Optimize` instance under the
      domain, per‑frame exclusivity and must‑link constraints.

IMPORTANT ASSUMPTIONS
———————————————————
* The caller ensures that *for every frame* the number of detections is
  ≤ `n_tracks`.  If that is violated the problem becomes UNSAT (because the
  frame‑wise “all‑different” constraint cannot be satisfied).
* Each detection **must** be assigned to some track.  Tracks that never
  receive a detection are omitted from the returned list.
* The embedding vectors (`Detection.feat`) are assumed L2‑normalised; if that
  is not true in your pipeline, normalise them first.
"""

from collections import defaultdict
from typing import Dict, List, Iterable, Tuple
from video_io import VideoInfo
import logging
from enum import Enum

import z3
import numpy as np

from common_types import *  # noqa: F401,F403 (Detection, Track, cosine_similarity ...)
from tracking import Tracker

MAX_TRACKS = 6
TIMEOUT_S = 30


def count_max_simultaneous_detections(
    detections: list[Detection],
) -> int:
    n_detections = defaultdict(int)
    for det in detections:
        n_detections[det.frame_idx] += 1
    return max(n_detections.values(), default=0)


class TimeoutException(Exception):
    """Raised when the Z3 solver times out."""
    pass


class UnsatisfiableException(Exception):
    """Raised when the Z3 solver finds the problem unsatisfiable."""
    pass


class _ComparisonResult(Enum):
    MATCH = "match"
    MAY_MATCH = "may_match"
    NO_MATCH = "no_match"


class DiscreteOptimizationTracker(Tracker):
    """Track objects by solving an assignment problem with *Z3*.

    The pipeline has two stages:
        (1) Greedy stitching of *obvious* continuation cases producing short
            track fragments (must‑link groups).
        (2) Global optimisation with Z3 honoring those must‑link groups while
            deciding how to interleave / merge fragments into up to `n_tracks`.

    Parameters
    ----------
    greedy_min_iou: float
        Minimum IoU considered a *clean* continuation.
    greedy_min_cosine_similarity: float
        Minimum cosine similarity for clean continuation.
    greddy_max_frame_distance: int
        Maximum allowed frame gap for greedy continuation (track becomes stale
        afterwards).
    greedy_min_iou_matches_single_track: float
        Floor IoU under which we declare NO_MATCH (helps uniqueness logic).
    """

    def __init__(
        self,
        greedy_min_iou: float = 0.8,
        greedy_min_cosine_similarity: float = 0.5,
        greddy_max_frame_distance: int = 5,
        greedy_min_iou_matches_single_track: float = 0.2,
    ):
        self.greedy_min_iou = greedy_min_iou
        self.greedy_min_cosine_similarity = greedy_min_cosine_similarity
        self.greedy_max_frame_distance = greddy_max_frame_distance
        self.min_iou_matches_single_track = greedy_min_iou_matches_single_track

    # ------------------------------------------------------------------
    # Greedy pre‑processing producing must‑link groups
    # ------------------------------------------------------------------

    def compare_track_to_detection(
        self,
        track: Track,
        detection: Detection,
    ) -> Tuple[_ComparisonResult, float]:
        iou = track.end().bbox.iou(detection.bbox)
        if iou < self.min_iou_matches_single_track:
            return (_ComparisonResult.NO_MATCH, 0)
        sim = cosine_similarity(track.end().feat, detection.feat)
        if iou >= self.greedy_min_iou and sim >= self.greedy_min_cosine_similarity:
            return (_ComparisonResult.MATCH, min(iou, sim))
        return (_ComparisonResult.MAY_MATCH, min(iou, sim))

    def _preprocess_detections(
        self,
        detections: list[Detection],
    ) -> List[Track]:
        """Greedily stiches detections onto tracks as long as both IOU and cosine similarity are high.

        Returns a list of *fragment tracks* — each becomes a must‑link group.
        """
        detections = sorted(detections, key=lambda d: d.frame_idx)
        stale_tracks: List[Track] = []
        active_tracks: List[Track] = []
        next_active_tracks: List[Track] = []
        for det in detections:
            clean_matches: list[tuple[Track, float]] = []
            mby_matches: list[tuple[Track, float]] = []
            for track in active_tracks:
                if track.end().frame_idx + self.greedy_max_frame_distance < det.frame_idx:
                    continue  # remove old tracks
                comparison_result, sim = self.compare_track_to_detection(track, det)
                if comparison_result == _ComparisonResult.MATCH:
                    clean_matches.append((track, sim))
                elif comparison_result == _ComparisonResult.MAY_MATCH:
                    mby_matches.append((track, sim))
                else:  # NO_MATCH
                    next_active_tracks.append(track)

            if len(clean_matches) == 1:
                track, _ = clean_matches[0]
                track.sorted_detections.append(det)
                next_active_tracks.append(track)
                for mby_track, _ in mby_matches:
                    next_active_tracks.append(mby_track)
            elif len(clean_matches) == 0 and len(mby_matches) == 1:
                track, _ = mby_matches[0]
                track.sorted_detections.append(det)
                next_active_tracks.append(track)
            else:
                new_track = Track(
                    track_id=len(active_tracks) + len(stale_tracks),
                    sorted_detections=[det],
                )
                next_active_tracks.append(new_track)

            # update stale vs active
            for track in active_tracks:
                if track not in next_active_tracks:
                    stale_tracks.append(track)
            active_tracks = next_active_tracks
            next_active_tracks = []
        return stale_tracks + active_tracks

    # ------------------------------------------------------------------
    # Core optimisation
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

        # Build a mapping from detection identity to its index after sorting.
        det_index = {id(det): idx for idx, det in enumerate(detections)}

        frame_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, det in enumerate(detections):
            frame_to_indices[det.frame_idx].append(idx)
        frames_sorted = sorted(frame_to_indices.keys())

        opt = z3.Optimize()
        opt.set("timeout", TIMEOUT_S * 1000)

        # Decision variables.
        assign: list[z3.IntNumRef] = [z3.Int(f"track_{idx}") for idx in range(num_det)]
        for var in assign:
            opt.add(z3.And(var >= 0, var < n_tracks))

        # Per-frame AllDifferent.
        for indices in frame_to_indices.values():
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    opt.add(assign[indices[a]] != assign[indices[b]])

        # Must-link equality constraints from pre-processing.
        if must_link_groups:
            seen = set()
            for group in must_link_groups:
                # Convert to indices; groups of size 1 add nothing.
                idxs = [det_index[id(d)] for d in group if id(d) in det_index]
                if len(idxs) < 2:
                    continue
                # Sanity: ensure no detection appears in two different groups.
                for i in idxs:
                    if i in seen:
                        raise ValueError("Detection appears in multiple must-link groups; groups must be disjoint.")
                    seen.add(i)
                base = assign[idxs[0]]
                for i in idxs[1:]:
                    opt.add(assign[i] == base)

        # Continuity cost (adjacent frames only).
        costs: List[z3.ArithRef] = []
        for f_idx in range(len(frames_sorted) - 1):
            f_curr, f_next = frames_sorted[f_idx], frames_sorted[f_idx + 1]
            for i in frame_to_indices[f_curr]:
                for j in frame_to_indices[f_next]:
                    cost_val = _pair_cost(detections[i], detections[j])
                    costs.append(
                        z3.If(assign[i] == assign[j], z3.RealVal(str(cost_val)), z3.RealVal("0"))
                    )
        total_cost = z3.Sum(costs) if costs else z3.RealVal("0")
        opt.minimize(total_cost)

        # Solve.
        result = opt.check()
        if result != z3.sat:
            if result == z3.unknown:
                logging.debug("Z3 solver timed out")
                raise TimeoutException("Z3 solver timed out")
            elif result == z3.unsat:
                logging.debug("Z3 solver found the problem unsatisfiable")
                raise UnsatisfiableException("Z3 solver found the problem unsatisfiable")
            else:
                raise RuntimeError(f"Unexpected Z3 solver status: {result}")

        model = opt.model()
        assign_vals = [model.evaluate(v).as_long() for v in assign]

        # Build Track objects.
        track_to_dets: Dict[int, List[Detection]] = defaultdict(list)
        for det, track_id in zip(detections, assign_vals):
            track_to_dets[track_id].append(det)
        tracks: List[Track] = []
        for track_id, dets in track_to_dets.items():
            dets_sorted = sorted(dets, key=lambda d: d.frame_idx)
            tracks.append(Track(track_id=track_id, sorted_detections=dets_sorted))
        return tracks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_detections(
        self,
        detections: list[Detection],
        video_properties: VideoInfo | None = None,
    ) -> list[Track]:
        logging.info(
            f"{'=' * 80} Running discrete optimization tracker with {len(detections)} detections {'=' * 80}"
        )
        max_detections = count_max_simultaneous_detections(detections)
        logging.info(f"Max simultaneous detections: {max_detections}")

        # Stage 1: greedy fragments (must‑link groups)
        fragments = self._preprocess_detections(detections)
        must_link_groups = [t.sorted_detections for t in fragments if len(t.sorted_detections) > 1]
        logging.info(
            f"Pre-processing produced {len(fragments)} fragments; {len(must_link_groups)} must-link groups (len>1)."
        )

        # Stage 2: global optimisation. (Currently we use max_detections as n_tracks.)
        n_tracks = max_detections
        logging.debug(f"Invoking solver with n_tracks={n_tracks}")
        solution_tracks = self._track_detections_inner(
            detections, n_tracks=n_tracks, must_link_groups=must_link_groups
        )
        logging.info(f"Solver produced {len(solution_tracks)} tracks")
        return solution_tracks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pair_cost(det_prev: "Detection", det_curr: "Detection") -> float:
    """Return *continuity cost* of linking *det_prev* → *det_curr*.

    Lower is better.  The formula is *(1 – IoU) + (1 – cosine_similarity)*.
    Values are clamped into [0, 2] for numerical stability.
    """
    iou = det_prev.bbox.iou(det_curr.bbox)
    cos_sim = cosine_similarity(det_prev.feat, det_curr.feat)
    iou = max(0.0, min(1.0, iou))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    cost = (1.0 - iou) + (1.0 - cos_sim)
    return float(max(0.0, min(2.0, cost)))
