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

import z3
import logging

from enum import Enum
from collections import defaultdict


from video_io import VideoInfo
from common_types import Detection, FrameIndex, Track, TrackId, cosine_similarity

MAX_TRACKS = 6
TIMEOUT_S = 30


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
    ):
        self.greedy_min_iou = greedy_min_iou
        self.greedy_min_cosine_similarity = greedy_min_cosine_similarity
        self.greedy_max_frame_distance = greedy_max_frame_distance
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
        # We only match against active tracks. Stracks become stale if they are too old or if they have been considered for a match but ot chosen

        # Sort detections by frame index to process them in order.
        sorted_frame_indices = sorted(detections_by_frame.keys())

        next_track_id = 1

        # these tracks have been detected and can be matched to further detections
        active_tracks: list[Track] = []
        # these tracks have been detected but can't match further detections
        stale_tracks: list[Track] = []
        stale_track_ids: set[TrackId] = set()

        for frame_idx in sorted_frame_indices:
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
                    track = clean_matches[0]
                    track.sorted_detections.append(detection)
                elif len(clean_matches) == 0 and len(mby_matches) == 1:
                    track = mby_matches[0]
                    track.sorted_detections.append(detection)
                else:
                    # no clear match found, create a new track for this detection
                    new_track = Track(track_id=next_track_id, sorted_detections=[detection])
                    active_tracks.append(new_track)
                    next_track_id += 1

                    # all clean and mby matches are stale because we couldn't cleanly match them
                    for track in clean_matches + mby_matches:
                        if track.track_id not in stale_track_ids:
                            stale_track_ids.add(track.track_id)
                            stale_tracks.append(track)

            # Too old tracks are stale
            for track in active_tracks:
                if track.end().frame_idx + self.greedy_max_frame_distance < frame_idx:
                    if track.track_id not in stale_track_ids:
                        stale_track_ids.add(track.track_id)
                        stale_tracks.append(track)

            # update stale / active tracks once per frame
            active_tracks = [track for track in active_tracks if track not in stale_tracks]

        return stale_tracks + active_tracks

    def _track_detections_inner(self, detections: list[Detection], n_tracks: int) -> list[Track]:
        if not detections:
            return []

        # Sort detections chronologically to simplify neighborhood logic.
        detections = sorted(detections, key=lambda d: d.frame_idx)
        num_det = len(detections)

        # -----------------------------------------------------------------
        # 1. Z3 variables
        # -----------------------------------------------------------------

        detection_to_track_assignments: list[z3.IntNumRef] = [z3.Int(f'track_{idx}') for idx in range(num_det)]

        opt = z3.Optimize()
        opt.set('timeout', TIMEOUT_S * 1000)

        # Track‑id domain: 0‥n_tracks‑1
        for track_assignment in detection_to_track_assignments:
            opt.add(z3.And(track_assignment >= 0, track_assignment < n_tracks))

        # -----------------------------------------------------------------
        # 2. Frame‑wise exclusivity constraints – AllDifferent per frame.
        #    Using pair‑wise inequality is more scalable than z3.Distinct when
        #    the per‑frame cardinality is high.
        # -----------------------------------------------------------------

        frame_index_to_assignment_indices: dict[FrameIndex, list[int]] = defaultdict(list)
        for idx, det in enumerate(detections):
            frame_index_to_assignment_indices[det.frame_idx].append(idx)

        for assignment_indices in frame_index_to_assignment_indices.values():
            # All assignments in a frame must be different
            # TODO: use z3.Distinct instead?
            opt.add(z3.Distinct(*[detection_to_track_assignments[i] for i in assignment_indices]))
            # for a in range(len(assignment_indices)):
            #     for b in range(a + 1, len(assignment_indices)):
            #         opt.add(
            #             detection_to_track_assignments[assignment_indices[a]]
            #             != detection_to_track_assignments[assignment_indices[b]]
            #         )

        # -----------------------------------------------------------------
        # 3. Continuity cost over successive frames.
        # -----------------------------------------------------------------

        costs: list[z3.ArithRef] = []

        # Build a quick mapping frame → indices to speed‑up the nested loops.
        frames_sorted = sorted(frame_index_to_assignment_indices.keys())

        for f_idx in range(len(frames_sorted) - 1):
            f_curr, f_next = frames_sorted[f_idx], frames_sorted[f_idx + 1]

            for i in frame_index_to_assignment_indices[f_curr]:
                for j in frame_index_to_assignment_indices[f_next]:
                    cost_val = _pair_cost(detections[i], detections[j])
                    costs.append(
                        z3.If(
                            detection_to_track_assignments[i] == detection_to_track_assignments[j],
                            z3.RealVal(str(cost_val)),
                            z3.RealVal('0'),
                        )
                    )

        total_cost = z3.Sum(costs) if costs else z3.RealVal('0')
        opt.minimize(total_cost)

        # -----------------------------------------------------------------
        # 4. Solve.
        # -----------------------------------------------------------------

        if opt.check() != z3.sat:
            if opt.check() == z3.unknown:
                logging.debug('Z3 solver timed out')
                raise TimeoutException('Z3 solver timed out')
            elif opt.check() == z3.unsat:
                logging.debug('Z3 solver found the problem unsatisfiable')
                raise UnsatisfiableException('Z3 solver found the problem unsatisfiable')
            else:
                raise RuntimeError(f'Unexpected Z3 solver status: {opt.check()}')

        model = opt.model()
        assign_vals = [model.evaluate(v).as_long() for v in detection_to_track_assignments]

        # -----------------------------------------------------------------
        # 5. Build Track objects in ascending time order.
        # -----------------------------------------------------------------

        track_to_dets: dict[int, list[Detection]] = defaultdict(list)
        for det, track_id in zip(detections, assign_vals):
            track_to_dets[track_id].append(det)

        tracks: list[Track] = []
        for track_id, dets in track_to_dets.items():
            dets_sorted = sorted(dets, key=lambda d: d.frame_idx)
            tracks.append(Track(track_id=track_id, sorted_detections=dets_sorted))

        return tracks

    def track_detections(self, detections: list[Detection], video_properties: VideoInfo | None = None) -> list[Track]:
        """Assign a *track id* to every detection and return the built tracks.

        The optimisation model minimises a continuity cost; see the module
        docstring for details.  If the problem is UNSAT an exception is raised.
        """
        logging.info(
            f'\n{"=" * 80}\nRunning discrete optimization tracker with {len(detections)} detections\n{"=" * 80}'
        )

        detections_by_frame: dict[FrameIndex, list[Detection]] = defaultdict(list)
        for det in detections:
            detections_by_frame[det.frame_idx].append(det)

        max_simultaneous_detections = max(len(detections) for detections in detections_by_frame.values())
        logging.info(f'Max simultaneous detections: {max_simultaneous_detections}')

        preprocessed_tracks = self._preprocess_detections(detections_by_frame)
        return preprocessed_tracks

        # TODO: increase detections to check if there might be a better solution
        # penalize new tracks
        max_tracks = max_simultaneous_detections + 1
        for n_tracks in range(max_simultaneous_detections, max_tracks):
            logging.debug(f'Trying to track with {n_tracks} tracks')
            try:
                sol = self._track_detections_inner(detections, n_tracks)
                logging.info(f'Tracking problem solved with {n_tracks} tracks, {len(sol)} tracks created')
                return sol
            except TimeoutException:
                raise ValueError('Z3 solver timed out; increase TIMEOUT_S.')
            except UnsatisfiableException:
                continue
        raise ValueError(
            f'Z3 solver could not find a solution with {max_tracks} tracks; '
            'increase `max_tracks` or check your input detections.'
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pair_cost(det_prev: Detection, det_curr: Detection) -> float:
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
