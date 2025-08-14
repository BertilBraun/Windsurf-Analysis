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
import logging

from ..settings import (
    GREEDY_MIN_COSINE_SIMILARITY,
    GREEDY_MIN_IOU_MATCHES_SINGLE_TRACK,
    GREEDY_SHORT_TRACK_MIN_FRAMES,
    MAX_OVERLAP_LENGTH_SECONDS,
)
from ..common_types import Track
from ..util.similarity_helpers import mean_embedding
from ..util.video_io import VideoInfo
from ..util.similarity_helpers import cosine_similarity


class GreedyTracker:
    def track(self, tracks: list[Track], video_properties: VideoInfo) -> list[Track]:
        """Greedily fuse tracklets whose average‑embedding similarity is highest.

        Parameters
        ----------
        tracks     : current list of **pre‑filtered** Track objects.
        sim_thresh : minimum cosine similarity required to merge.
        max_gap    : max number of frames allowed between two adjacent tracklets.
        verbose    : print merge operations if *True*.
        """
        logging.info(f'{"=" * 30} Running greedy tracker with {len(tracks)} tracks {"=" * 30}')

        working = list(tracks)  # copy

        max_gap = video_properties.fps * MAX_OVERLAP_LENGTH_SECONDS

        while True:
            n = len(working)
            if n < 2:
                break

            # Pre‑compute average embeddings
            avg_emb = [mean_embedding(t) for t in working]

            best_i, best_j, best_sim = None, None, -1.0
            # Evaluate all unordered pairs
            for i in range(n):
                for j in range(i + 1, n):
                    if not _can_merge(working[i], working[j], max_gap=max_gap):
                        continue
                    sim = cosine_similarity(avg_emb[i], avg_emb[j])
                    # TODO prefer to merge longer tracks?
                    # longest_track_length = max(len(t.sorted_detections) for t in working)
                    # (len(working[i].sorted_detections) + len(working[j].sorted_detections)) / (2 * longest_track_length)
                    if sim > best_sim:
                        best_i, best_j, best_sim = i, j, sim

            # Stopping condition
            if best_sim < GREEDY_MIN_COSINE_SIMILARITY:
                break

            assert best_i is not None and best_j is not None
            assert 0 <= best_i < n and 0 <= best_j < n

            # Merge the best pair
            new_track = _merge_tracks(working[best_i], working[best_j])

            # Replace indices i & j with t_new
            working = [t for k, t in enumerate(working) if k not in (best_i, best_j)]
            working.append(new_track)

        return working


def _can_merge(t1: Track, t2: Track, *, max_gap: int) -> bool:
    """Return *True* if tracks' frame sets are disjoint and temporal order is OK."""
    frames1, frames2 = set(t1.detections_by_frame.keys()), set(t2.detections_by_frame.keys())
    if frames1 & frames2:
        return False  # overlapping detections – ambiguous

    if len(frames1) < GREEDY_SHORT_TRACK_MIN_FRAMES or len(frames2) < GREEDY_SHORT_TRACK_MIN_FRAMES:
        return False  # Very short tracks are merged in the DiscreteOptTracker

    start1, end1 = min(frames1), max(frames1)
    start2, end2 = min(frames2), max(frames2)

    # t2 comes after t1
    if end1 < start2 and (start2 - end1) <= max_gap:
        iou_end1_start2 = t1.detections_by_frame[end1].bbox.iou(t2.detections_by_frame[start2].bbox)
        return iou_end1_start2 > GREEDY_MIN_IOU_MATCHES_SINGLE_TRACK

    # or t1 comes after t2
    if end2 < start1 and (start1 - end2) <= max_gap:
        iou_end2_start1 = t2.detections_by_frame[end2].bbox.iou(t1.detections_by_frame[start1].bbox)
        return iou_end2_start1 > GREEDY_MIN_IOU_MATCHES_SINGLE_TRACK

    # or one inside a gap of the other (but still non‑overlapping)
    if start1 < start2 and end2 < end1:
        return True
    if start2 < start1 and end1 < end2:
        return True

    return False


def _merge_tracks(t1: Track, t2: Track) -> Track:
    """Return a **new** Track with detections = union(t1, t2)."""
    new_detections = list(sorted(t1.sorted_detections + t2.sorted_detections, key=lambda d: d.frame_idx))
    # keep the lower track id for reproducibility; change if you prefer a fresh id
    new_id = min(t1.track_id, t2.track_id)
    return Track(track_id=new_id, sorted_detections=new_detections)
