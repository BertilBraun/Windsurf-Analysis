# track_filter.py
"""Utility helpers to cull *spurious* tracks from a multi‑object‑tracking result.

The heuristic now removes **short & spatially isolated** tracks using a *two‑step
check*:

1. **Duration** — candidate track has fewer than *min_frames* detections.
2. **Inflated‑bbox overlap** — for every detection in that short track we look
   at *long* tracks (≥ *min_frames*). We enlarge each long‑track detection’s
   bbox by an *enlarge_factor* (default **1.5×** on width & height, centred),
   then ask if the candidate’s bbox intersects any of those inflated boxes.
   *   **Yes** → it’s near a real track → **keep** the candidate.
   *   **No**  → it’s an isolated blip → **remove**.

You can still enforce IoU or centre‑distance thresholds in addition to—or
instead of—the inflation test.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple


from common_types import Detection, FrameIndex, Track, TrackId


def filter_non_surfers_from_tracks(
    tracks: Sequence[Track],
    *,
    min_frames: int = 5,
    enlarge_factor: float = 1.5,
    iou_thresh: float = 0.0,
) -> Tuple[List[Track], List[Track]]:
    """Return *(kept, removed)* after pruning short, isolated tracks.

    A track is *removed* when **both** conditions hold:
    1. It has ``len(track) < min_frames`` detections.
    2. None of its detections intersect the *inflated* bounding boxes of any
       *long* track (≥ *min_frames*) in the same frame.

    Parameters
    ----------
    min_frames      : int
        Threshold separating *long* vs. *short* tracks.
    enlarge_factor  : float
        Scale factor applied to long‑track bounding boxes before overlap tests.
    iou_thresh      : float
        Optional stricter test – require IoU ≥ *iou_thresh* with the inflated
        box.  Set ≤0 to disable.
    distance_thresh : float | None
        Additional centre‑distance cutoff (pixels).  Ignored if *None*.
    """
    if min_frames <= 0:
        raise ValueError('min_frames must be positive.')
    if enlarge_factor <= 0:
        raise ValueError('enlarge_factor must be positive.')

    # Build frame → list[(track_id, det)] maps, separated into long vs. short.
    frame_map_long: defaultdict[FrameIndex, list[tuple[TrackId, Detection]]] = defaultdict(list)
    frame_map_short: defaultdict[FrameIndex, list[tuple[TrackId, Detection]]] = defaultdict(list)

    for track in tracks:
        target_map = frame_map_long if len(track.sorted_detections) >= min_frames else frame_map_short
        for short_detection in track.sorted_detections:
            target_map[short_detection.frame_idx].append((track.track_id, short_detection))

    kept: List[Track] = []
    removed: List[Track] = []

    for track in tracks:
        if len(track.sorted_detections) >= min_frames:
            kept.append(track)  # always keep long tracks
            continue

        # Evaluate isolation against LONG tracks only
        isolated = True
        for short_detection in track.sorted_detections:
            for _long_track_id, long_detection in frame_map_long[short_detection.frame_idx]:
                if _detections_overlap(
                    short_detection,
                    long_detection,
                    enlarge_factor=enlarge_factor,
                    iou_thresh=iou_thresh,
                ):
                    isolated = False
                    break
            if not isolated:
                break

        (removed if isolated else kept).append(track)

    return kept, removed


def _detections_overlap(
    short_detection: Detection,
    long_detection: Detection,
    *,
    enlarge_factor: float,
    iou_thresh: float,
) -> bool:
    """Does *short_detection* intersect the **inflated** bbox of *long_detection*?"""
    long_bbox_inflated = long_detection.bbox.scale(enlarge_factor)

    # Simple bbox‑intersection quick exit
    if long_bbox_inflated.overlaps(short_detection.bbox):
        if iou_thresh <= 0:
            return True  # intersection alone is sufficient

        # IoU check
        iou = short_detection.bbox.iou(long_bbox_inflated)
        if iou >= iou_thresh:
            return True

    return False
