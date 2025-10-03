from __future__ import annotations

import logging
from enum import Enum
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

from server.inference.bot_sort.cmc import CMC
from server.inference.src.util.similarity_helpers import Embedding, HistogramEmbedding
from server.inference.src.visualization.stabilize import Transform
from ...util.video_io import VideoInfo
from ...common_types import Detection, FrameIndex, Track, TrackId
from server.inference.bot_sort.kalman_filter import KFState


class _ComparisonResult(Enum):
    MATCH = 'match'
    MAY_MATCH = 'may_match'
    NO_MATCH = 'no_match'


# ───────────────────────────── main class ───────────────────────────── #


class GreedyTrackStitcher:
    """
    Greedy pre-stitcher with original matching orchestration, but using:
      • Motion: KF + forward GMC deltas (prev→curr) → position-only Mahalanobis d²
      • Appearance: χ² distance on L1-normalized L*a*b* histograms (with EMA)

    Decision:
      MATCH      if (d² ≤ gate_strict) and (χ² ≤ chi2_strict)
      MAY_MATCH  if (d² ≤ gate_loose)  and (χ² ≤ chi2_loose)
      NO_MATCH   otherwise

    Notes:
      - Isolation heuristic intentionally disabled (can be re-added if needed).
      - KF predictions are cached per track via KFState to avoid recomputation.
    """

    def __init__(
        self,
        *,
        gate_strict_d2: float = 5.9915,  # χ²(2, 0.95)
        gate_loose_d2: float = 9.2103,  # χ²(2, 0.99)
        chi2_strict: float = 0.20,  # tune on your data
        chi2_loose: float = 0.35,  # tune on your data
        ema_alpha: float = 0.9,  # appearance EMA smoothing
        max_frame_distance: int = 5,  # stale cutoff (frames)
    ):
        self.gate_strict_d2 = float(gate_strict_d2)
        self.gate_loose_d2 = float(gate_loose_d2)
        self.chi2_strict = float(chi2_strict)
        self.chi2_loose = float(chi2_loose)
        self.ema_alpha = float(ema_alpha)
        self.max_frame_distance = int(max_frame_distance)

        # Per-track state
        self._kf: Dict[TrackId, KFState] = {}
        self._ema: Dict[TrackId, Embedding] = {}

    # ─────────────────────────── public API ─────────────────────────── #

    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        """Greedily stitches single-detection tracks into longer tracks (old orchestration, new compare)."""
        logging.info(f'{"=" * 30} Running greedy preprocessor with {len(tracks)} tracks {"=" * 30}')

        # Build per-frame detections from single-detection inputs
        detections_by_frame: Dict[FrameIndex, List[Detection]] = defaultdict(list)
        for track in tracks:
            assert len(track.sorted_detections) == 1, 'Greedy preprocessor only supports single-detection tracks'
            for det in track.sorted_detections:
                detections_by_frame[det.frame_idx].append(det)

        frames: List[int] = sorted(int(f) for f in detections_by_frame.keys())

        cmc = CMC(transforms)

        next_track_id: int = 1

        # Active = can be extended; Stale = finalized and kept for output
        active_tracks: List[Track] = []
        stale_tracks: List[Track] = []
        stale_track_ids: set[TrackId] = set()

        for frame_idx in frames:
            matches_this_frame: List[tuple[Track, Detection]] = []

            # ── propose matches greedily per detection ──
            for detection in detections_by_frame[frame_idx]:
                clean_matches: List[Track] = []
                mby_matches: List[Track] = []

                for track in active_tracks:
                    result = self._compare_detection_to_track(track, detection, cmc=cmc)
                    if result == _ComparisonResult.MATCH:
                        clean_matches.append(track)
                    elif result == _ComparisonResult.MAY_MATCH:
                        mby_matches.append(track)

                if len(clean_matches) == 1:
                    matches_this_frame.append((clean_matches[0], detection))
                elif len(clean_matches) == 0 and len(mby_matches) == 1:
                    matches_this_frame.append((mby_matches[0], detection))
                else:
                    # No clear match → new track candidate for this detection
                    new_track = Track(track_id=next_track_id, sorted_detections=[])
                    matches_this_frame.append((new_track, detection))
                    next_track_id += 1

                    # All tracks that "almost matched" become stale
                    for track in clean_matches + mby_matches:
                        if track.track_id not in stale_track_ids:
                            stale_track_ids.add(track.track_id)
                            stale_tracks.append(track)
                            # cleanup state
                            self._kf.pop(track.track_id, None)
                            self._ema.pop(track.track_id, None)

            # ── resolve conflicts per track id ──
            detections_per_track: Dict[int, List[Detection]] = defaultdict(list)
            tracks_per_track_id: Dict[int, Track] = {}
            for track, detection in matches_this_frame:
                detections_per_track[track.track_id].append(detection)
                tracks_per_track_id[track.track_id] = track

            for track_id, detections in detections_per_track.items():
                track = tracks_per_track_id[track_id]
                if len(detections) > 1:
                    # conflicting attachments → stale original, spawn new tracks
                    if track.track_id not in stale_track_ids:
                        stale_track_ids.add(track.track_id)
                        stale_tracks.append(track)
                        self._kf.pop(track.track_id, None)
                        self._ema.pop(track.track_id, None)
                    for det in detections:
                        new_track = Track(track_id=next_track_id, sorted_detections=[det])
                        next_track_id += 1
                        active_tracks.append(new_track)
                        # init per-track state
                        self._kf[new_track.track_id] = KFState.init(new_track)
                        self._ema[new_track.track_id] = det.embedding
                else:
                    # single extension
                    det = detections[0]
                    track.sorted_detections.append(det)
                    if track not in active_tracks:
                        active_tracks.append(track)
                    # init if needed, then update KF + EMA
                    if track.track_id not in self._kf:
                        self._kf[track.track_id] = KFState.init(track)
                        self._ema[track.track_id] = track.start.embedding
                    self._kf[track.track_id] = self._kf[track.track_id].update_to_det(det, cmc)
                    old_ema = self._ema.get(track.track_id, track.start.embedding)
                    self._ema[track.track_id] = old_ema.interpolate(det.embedding, self.ema_alpha)

            # ── age out stale tracks (too far behind current frame) ──
            for track in list(active_tracks):
                if track.end.frame_idx + self.max_frame_distance < frame_idx:
                    if track.track_id not in stale_track_ids:
                        stale_track_ids.add(track.track_id)
                        stale_tracks.append(track)
                        self._kf.pop(track.track_id, None)
                        self._ema.pop(track.track_id, None)

            # keep only non-stale active tracks
            active_tracks = [track for track in active_tracks if track.track_id not in stale_track_ids]

        return stale_tracks + active_tracks

    # ───────────────────────────── scoring ───────────────────────────── #

    def _compare_detection_to_track(self, track: Track, detection: Detection, cmc: CMC) -> _ComparisonResult:
        """
        Score motion (Mahalanobis d² on [cx,cy]) + appearance (χ² on L1-hist EMA).
        """
        # Early guard on excessive frame gap (keeps candidate set small)
        gap = detection.frame_idx - track.end.frame_idx
        if gap <= 0 or gap > self.max_frame_distance:
            return _ComparisonResult.NO_MATCH

        tid = track.track_id

        # Ensure KF state
        st = self._kf.get(tid)
        if st is None:
            st = self._kf[tid] = KFState.init(track)

        # Predict to current frame (cached inside KFState)
        pred = st.predict_to(detection.frame_idx, cmc)

        # Motion gating: squared Mahalanobis distance (df=2)
        d2 = pred.gating_distance(detection.bbox.center_wh)

        # Appearance: χ² on L1-normalized histograms with EMA
        ema = self._ema.get(tid, track.start.embedding)
        assert isinstance(ema, HistogramEmbedding)
        chi2 = ema.distance(detection.embedding)

        # Decision
        if d2 <= self.gate_strict_d2 and chi2 <= self.chi2_strict:
            return _ComparisonResult.MATCH

        if d2 <= self.gate_loose_d2 and chi2 <= self.chi2_loose:
            # Isolation logic intentionally disabled for now.
            # If needed later, reintroduce a tie-break under the loose gate.
            return _ComparisonResult.MAY_MATCH

        return _ComparisonResult.NO_MATCH
