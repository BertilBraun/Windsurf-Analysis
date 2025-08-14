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

import numpy as np
from typing import List, Sequence, Tuple

from ...common_types import Track, TrackId
from ...util.similarity_helpers import cosine_similarity, mean_embedding


def filter_non_surfers_from_tracks(
    tracks: Sequence[Track],
    *,
    min_frames: int = 5,
    similarity_thresh: float = 0.2,
) -> Tuple[List[Track], List[Track]]:
    """Return *(kept, removed)* after pruning short, isolated tracks.

    A track is *removed* when **both** conditions hold:
    1. It has ``len(track) < min_frames`` detections.
    2. Its average embedding cosine similarity with any *long* track (≥ *min_frames*) is less than *similarity_thresh*.

    Parameters
    ----------
    min_frames      : int
        Threshold separating *long* vs. *short* tracks.
    similarity_thresh : float
        Minimum cosine similarity between average embeddings of tracks to consider merging.
    """
    if min_frames <= 0:
        raise ValueError('min_frames must be positive.')
    if similarity_thresh <= 0:
        raise ValueError('similarity_thresh must be positive.')

    kept: List[Track] = []
    removed: List[Track] = []

    track_id_to_average_embedding: dict[TrackId, np.ndarray] = {}
    for track in tracks:
        if len(track.sorted_detections) >= min_frames:
            track_id_to_average_embedding[track.track_id] = mean_embedding(track)

    for track in tracks:
        if len(track.sorted_detections) >= min_frames:
            kept.append(track)  # always keep long tracks
            continue

        short_track_average_embedding = mean_embedding(track)

        for long_track_average_embedding in track_id_to_average_embedding.values():
            if cosine_similarity(short_track_average_embedding, long_track_average_embedding) > similarity_thresh:
                kept.append(track)
                break
        else:
            removed.append(track)

    return kept, removed
