from __future__ import annotations

"""Discrete‑optimization based multi‑object tracker using Z3.

This implementation follows the design sketched in the conversation:
    • A *decision variable* per detection selects which track the detection
      belongs to (domain: 0‥n_tracks‑1).
    • *Frame‑level exclusivity*: no two detections that appear in the same
      frame may share the same track id. This is encoded with pair‑wise
      inequality constraints.
    • *Cost function*: when two detections in consecutive frames belong to the
      same track, the model accrues a cost equal to

            (1 – IoU(bbox_prev, bbox_curr))
          + (1 – cosine_similarity(feat_prev, feat_curr))

      Lower cost ⇒ nicer continuity. Only **adjacent** frames are considered
      to keep the formulation compact.  Supporting arbitrary frame gaps is
      possible but requires extra bookkeeping variables.

    • The total cost is minimised with a Z3 `Optimize` instance.

IMPORTANT ASSUMPTIONS
———————————————————
* The caller ensures that *for every frame* the number of detections is
  ≤ `n_tracks`.  If that is violated the problem becomes UNSAT (because the
  frame‑wise “all‑different” constraint cannot be satisfied).
* Each detection **must** be assigned to some track.  Tracks that never
  receive a detection are omitted from the returned list.
* The embedding vectors (`Detection.feat`) are assumed L2‑normalised; if that
  is not true in your pipeline, normalise them first.
"""

from collections import defaultdict
from typing import Dict, List
from video_io import VideoInfo
import logging

import z3
import numpy as np

from common_types import *
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


class DiscreteOptimizationTracker(Tracker):
    """Track objects by solving an assignment problem with *Z3*.

    Parameters
    ----------
    n_tracks:
        Maximum number of tracks that may exist in the scene.  Must be ≥ the
        maximum number of detections observed in any single frame.
    """

    def _track_detections_inner(
            self,
            detections: list[Detection],
            n_tracks: int,
    ) -> List[Track]:
        if not detections:
            return []

        # Sort detections chronologically to simplify neighbourhood logic.
        detections = sorted(detections, key=lambda d: d.frame_idx)
        num_det = len(detections)

        # -----------------------------------------------------------------
        # 1. Z3 variables
        # -----------------------------------------------------------------

        assign: list[z3.IntNumRef] = [z3.Int(f"track_{idx}") for idx in range(num_det)]

        opt = z3.Optimize()
        opt.set("timeout", TIMEOUT_S * 1000)


        # Track‑id domain: 0‥n_tracks‑1
        for var in assign:
            opt.add(z3.And(var >= 0, var < n_tracks))

        # -----------------------------------------------------------------
        # 2. Frame‑wise exclusivity constraints – AllDifferent per frame.
        #    Using pair‑wise inequality is more scalable than z3.Distinct when
        #    the per‑frame cardinality is high.
        # -----------------------------------------------------------------

        frame_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, det in enumerate(detections):
            frame_to_indices[det.frame_idx].append(idx)

        for indices in frame_to_indices.values():
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    opt.add(assign[indices[a]] != assign[indices[b]])

        # -----------------------------------------------------------------
        # 3. Continuity cost over successive frames.
        # -----------------------------------------------------------------

        costs: List[z3.ArithRef] = []

        # Build a quick mapping frame → indices to speed‑up the nested loops.
        frames_sorted = sorted(frame_to_indices.keys())

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

        # -----------------------------------------------------------------
        # 4. Solve.
        # -----------------------------------------------------------------

        if opt.check() != z3.sat:
            if opt.check() == z3.unknown:
                logging.debug("Z3 solver timed out")
                raise TimeoutException("Z3 solver timed out")
            elif opt.check() == z3.unsat:
                logging.debug("Z3 solver found the problem unsatisfiable")
                raise UnsatisfiableException("Z3 solver found the problem unsatisfiable")
            else:
                raise RuntimeError(
                    f"Unexpected Z3 solver status: {opt.check()}"
                )

        model = opt.model()
        assign_vals = [model.evaluate(v).as_long() for v in assign]

        # -----------------------------------------------------------------
        # 5. Build Track objects in ascending time order.
        # -----------------------------------------------------------------

        track_to_dets: Dict[int, List["Detection"]] = defaultdict(list)
        for det, track_id in zip(detections, assign_vals):
            track_to_dets[track_id].append(det)

        tracks: List["Track"] = []
        for track_id, dets in track_to_dets.items():
            dets_sorted = sorted(dets, key=lambda d: d.frame_idx)
            tracks.append(Track(track_id=track_id, sorted_detections=dets_sorted))

        return tracks

    def track_detections(
        self,
        detections: list[Detection],
        video_properties: VideoInfo | None = None,
    ) -> list[Track]:
        """Assign a *track id* to every detection and return the built tracks.

        The optimisation model minimises a continuity cost; see the module
        docstring for details.  If the problem is UNSAT an exception is raised.
        """
        logging.info(
            f"\n{'=' * 80}\nRunning discrete optimization tracker with {len(detections)} detections\n{'=' * 80}"
        )
        max_detections = count_max_simultaneous_detections(detections)
        logging.info(f"Max simultaneous detections: {max_detections}")

        # TODO: increase detections to check if there might be a better solution
        # penalize new tracks
        for n_tracks in range(max_detections, max_detections + 1):
            logging.debug(f"Trying to track with {n_tracks} tracks")
            try:
                sol = self._track_detections_inner(detections, n_tracks)
                logging.info(
                    f"Tracking problem solved with {n_tracks} tracks, {len(sol)} tracks created"
                )
                return sol
            except TimeoutException:
                raise ValueError("Z3 solver timed out; increase TIMEOUT_S.")
            except UnsatisfiableException:
                continue
        raise ValueError(
            f"Z3 solver could not find a solution with {MAX_TRACKS} tracks; "
            "increase MAX_TRACKS or check your input detections."
        )

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pair_cost(det_prev: "Detection", det_curr: "Detection") -> float:
    """Return *continuity cost* of linking *det_prev* → *det_curr*.

    Lower is better.  The formula is *(1 – IoU) + (1 – cosine_similarity)*.
    Values are clamped into [0, 2] for numerical stability.
    """

    iou = det_prev.bbox.iou(det_curr.bbox)
    cos_sim = cosine_similarity(det_prev.feat, det_curr.feat)

    # Clamp to legal range just in case floating point noise pushes us a bit out.
    iou = max(0.0, min(1.0, iou))
    cos_sim = max(-1.0, min(1.0, cos_sim))

    cost = (1.0 - iou) + (1.0 - cos_sim)
    return float(max(0.0, min(2.0, cost)))
