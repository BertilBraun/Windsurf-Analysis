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
from typing import List

from ...common_types import Track, TrackId
from ...util.video_io import VideoInfo
from ...util.similarity_helpers import Embedding


class FilterNonSurfers:
    def __init__(self, min_frames: int, similarity_thresh: float):
        self.min_frames = min_frames
        self.similarity_thresh = similarity_thresh

    def track(self, tracks: list[Track], video_properties: VideoInfo) -> list[Track]:
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
        if self.min_frames <= 0:
            raise ValueError('min_frames must be positive.')
        if self.similarity_thresh <= 0:
            raise ValueError('similarity_thresh must be positive.')

        kept: List[Track] = []

        track_id_to_average_embedding: dict[TrackId, Embedding] = {}
        for track in tracks:
            if len(track.sorted_detections) >= self.min_frames:
                track_id_to_average_embedding[track.track_id] = track.mean_embedding()

        for track in tracks:
            if len(track.sorted_detections) >= self.min_frames:
                kept.append(track)  # always keep long tracks
                continue

            short_track_average_embedding = track.mean_embedding()

            for long_track_average_embedding in track_id_to_average_embedding.values():
                if short_track_average_embedding.distance(long_track_average_embedding) > self.similarity_thresh:
                    kept.append(track)
                    break

        return kept
