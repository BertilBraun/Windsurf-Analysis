# track_stitch.py
"""Greedy stitching of tracklets by cosine‐similarity.

The algorithm merges two tracks at a time, always picking the pair whose
**average‑embedding cosine similarity is currently the highest**, provided their
frame ranges do **not overlap** (they may touch or leave a gap ≤ *max_gap*).  If
one track’s time span sits entirely *inside* a gap of the other, the detections
are inserted in the middle – yielding a single, chronologically ordered track.

Stop when the best remaining similarity drops below *sim_thresh* or only one
track remains.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from tracking.preprocessing.greedy_preprocessor import GreedyPreprocessor
from video_io import VideoInfo

from common_types import Detection, Track, cosine_similarity


def _average_embedding(track: Track) -> np.ndarray:
    return np.mean([d.embedding for d in track.sorted_detections], axis=0)


def _can_merge(t1: Track, t2: Track, *, max_gap: int) -> bool:
    """Return *True* if tracks' frame sets are disjoint and temporal order is OK."""
    frames1, frames2 = set(t1.detections_by_frame.keys()), set(t2.detections_by_frame.keys())
    if frames1 & frames2:
        return False  # overlapping detections – ambiguous

    if len(frames1) < 10 or len(frames2) < 10:
        return False

    start1, end1 = min(frames1), max(frames1)
    start2, end2 = min(frames2), max(frames2)

    # iou between start1 end2 and start2 end1
    iou_start1_end2 = t1.detections_by_frame[start1].bbox.iou(t2.detections_by_frame[end2].bbox)  # 1 starts after 2
    iou_start2_end1 = t2.detections_by_frame[start2].bbox.iou(t1.detections_by_frame[end1].bbox)  # 2 starts after 1

    # t2 comes after t1
    if start2 > end1 and (start2 - end1) <= max_gap and iou_start2_end1 > 0.5:
        return True
    # or t1 comes after t2
    if start1 > end2 and (start1 - end2) <= max_gap and iou_start1_end2 > 0.5:
        return True

    # or one inside a gap of the other (but still non‑overlapping)
    if start1 < start2 < end1 and end2 > end1:
        if start2 - 1 in frames1:
            iou_start = t1.detections_by_frame[start2 - 1].bbox.iou(t2.detections_by_frame[start2].bbox)
        else:
            iou_start = 1.0  # TODO think about this
        if end2 + 1 in frames1:
            iou_end = t1.detections_by_frame[end2].bbox.iou(t2.detections_by_frame[end2 + 1].bbox)
        else:
            iou_end = 1.0  # TODO think about this
        if iou_start > 0.5 and iou_end > 0.5:
            return True  # insert t2 tail after t1 body
    if start2 < start1 < end2 and end1 > end2:
        if start1 - 1 in frames2:
            iou_start = t2.detections_by_frame[start1 - 1].bbox.iou(t1.detections_by_frame[start1].bbox)
        else:
            iou_start = 1.0  # TODO think about this
        if end1 + 1 in frames2:
            iou_end = t2.detections_by_frame[end1].bbox.iou(t1.detections_by_frame[end1 + 1].bbox)
        else:
            iou_end = 1.0  # TODO think about this
        if iou_start > 0.5 and iou_end > 0.5:
            return True  # insert t1 tail after t2 body

    return False


def _merge_tracks(t1: Track, t2: Track) -> Track:
    """Return a **new** Track with detections = union(t1, t2)."""
    new_dets = sorted(t1.sorted_detections + t2.sorted_detections, key=lambda d: d.frame_idx)
    # keep the lower track id for reproducibility; change if you prefer a fresh id
    new_id = t1.track_id if (t1.track_id is not None and t1.track_id <= (t2.track_id or 1e9)) else t2.track_id
    return Track(track_id=new_id, sorted_detections=new_dets)


###############################################################################
# Greedy stitching routine
###############################################################################


def greedy_stitch_tracks(
    tracks: Sequence[Track],
    *,
    sim_thresh: float = 0.7,
    max_gap: int = 30 * 10,
    verbose: bool = False,
) -> List[Track]:
    """Greedily fuse tracklets whose average‑embedding similarity is highest.

    Parameters
    ----------
    tracks     : current list of **pre‑filtered** Track objects.
    sim_thresh : minimum cosine similarity required to merge.
    max_gap    : max number of frames allowed between two adjacent tracklets.
    verbose    : print merge operations if *True*.
    """
    working = list(tracks)  # copy

    # TODO cap the number of frames to use for average embedding from the start and end of the track depending

    while True:
        n = len(working)
        if n < 2:
            break

        # Pre‑compute average embeddings
        avg_emb = [_average_embedding(t) for t in working]

        longest_track_length = max(len(t.sorted_detections) for t in working)

        best_i, best_j, best_sim = None, None, -1.0
        # Evaluate all unordered pairs
        for i in range(n):
            for j in range(i + 1, n):
                if not _can_merge(working[i], working[j], max_gap=max_gap):
                    continue
                sim = cosine_similarity(avg_emb[i], avg_emb[j]) + (
                    len(working[i].sorted_detections) + len(working[j].sorted_detections)
                ) / (2 * longest_track_length)
                if sim > best_sim:
                    best_i, best_j, best_sim = i, j, sim

        # Stopping condition
        if best_sim < sim_thresh:
            if verbose:
                print(f'Stopping – best similarity {best_sim:.3f} below threshold {sim_thresh}')
            break

        assert best_i is not None and best_j is not None
        assert 0 <= best_i < n and 0 <= best_j < n

        # Merge the best pair
        t_new = _merge_tracks(working[best_i], working[best_j])
        if verbose:
            id_i, id_j = working[best_i].track_id, working[best_j].track_id
            print(f'Merging tracks {id_i} & {id_j} (sim={best_sim:.3f}) → new id {t_new.track_id}')

        # Replace indices i & j with t_new
        new_list = [t for k, t in enumerate(working) if k not in (best_i, best_j)]
        new_list.append(t_new)
        working = new_list

    return working


class GreedyTracker:
    def track_detections(self, detections: list[Detection], video_properties: VideoInfo | None = None) -> list[Track]:
        if not detections:
            return []

        fragments = GreedyPreprocessor().track_detections(detections, video_properties)
        return greedy_stitch_tracks(fragments)
