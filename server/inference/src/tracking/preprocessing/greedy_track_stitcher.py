from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import DefaultDict, Dict, Iterable, List, Tuple
from collections import defaultdict

import numpy as np

from server.inference.src.settings import EPS
from server.inference.src.visualization.stabilize import Transform
from ...util.video_io import VideoInfo
from ...common_types import BoundingBox, Detection, FrameIndex, Point, Track, TrackId
from server.inference.bot_sort.kalman_filter import KalmanFilter


def chi2_dist(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    """Symmetric χ² distance for L*a*b* histogram embeddings (assumed L1-normalized)."""
    num = (p - q) ** 2
    den = p + q + eps
    return 0.5 * float((num / den).sum())


class _ComparisonResult(Enum):
    MATCH = 'match'
    MAY_MATCH = 'may_match'
    NO_MATCH = 'no_match'


@dataclass
class _KFState:
    mean: np.ndarray  # (8,)
    cov: np.ndarray  # (8,8)
    last_frame: int


# ────────────────────────────── main class ────────────────────────────── #


class GreedyTrackStitcher:
    """
    Greedy pre-stitcher using:
      • Motion: KF + forward GMC deltas (prev→curr) → position-only Mahalanobis d²
      • Appearance: χ² distance on L*a*b* hist embeddings (L1-normalized)

    Decision:
      MATCH      if (d² ≤ gate_strict) and (χ² ≤ chi2_strict)
      MAY_MATCH  if (d² ≤ gate_loose)  and (χ² ≤ chi2_loose) and detection is isolated
      NO_MATCH   otherwise
    """

    def __init__(
        self,
        gate_strict_d2: float = 5.9915,  # χ²(2, 0.95)
        gate_loose_d2: float = 9.2103,  # χ²(2, 0.99)
        chi2_strict: float = 0.5,  # tune on data
        chi2_loose: float = 0.95,  # tune on data
        iso_center_mul: float = 0.01,  # isolation: center dist > iso_center_mul * det.width
        iso_iou_max: float = 0.78,  # and IoU to others <= iso_iou_max
        max_frame_distance: int = 5,  # stale cutoff (frames)
        ema_alpha: float = 0.9,  # appearance EMA smoothing
    ):
        self.gate_strict_d2 = float(gate_strict_d2)
        self.gate_loose_d2 = float(gate_loose_d2)
        self.chi2_strict = float(chi2_strict)
        self.chi2_loose = float(chi2_loose)
        self.iso_center_mul = float(iso_center_mul)
        self.iso_iou_max = float(iso_iou_max)
        self.max_frame_distance = int(max_frame_distance)
        self.ema_alpha = float(ema_alpha)

        self._kf = KalmanFilter()

    # ─────────────────────────── public API ─────────────────────────── #

    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        """Greedily stitches single-detection tracks into longer tracks."""
        logging.info(f'{"=" * 28} Greedy preprocessor: {len(tracks)} inputs {"=" * 28}')
        dets_by_frame = self._group_single_detections_by_frame(tracks)
        frames = sorted(dets_by_frame.keys())
        gmc = {int(t.frame_idx): t for t in transforms}  # per-frame forward delta (f-1 → f)

        active: List[Track] = []
        stale: List[Track] = []
        stale_ids: set[TrackId] = set()
        kf_state: Dict[TrackId, _KFState] = {}
        next_tid = 1

        for f in frames:
            candidates = dets_by_frame[f]
            matches = self._collect_matches_for_frame(f, candidates, active, kf_state, gmc)

            # resolve per-track conflicts & update states
            new_tracks, newly_staled, next_tid = self._apply_matches(matches, active, kf_state, next_tid, gmc)
            stale.extend(newly_staled)
            stale_ids.update(t.track_id for t in newly_staled)
            active = new_tracks

            # age out
            aged_out = [t for t in active if (t.end.frame_idx + self.max_frame_distance) < f]
            for t in aged_out:
                if t.track_id not in stale_ids:
                    stale_ids.add(t.track_id)
                    stale.append(t)
            active = [t for t in active if t.track_id not in stale_ids]

        return stale + active

    # ─────────────────────── per-frame orchestration ─────────────────────── #

    def _collect_matches_for_frame(
        self,
        frame: int,
        detections: List[Detection],
        active_tracks: List[Track],
        kf_state: Dict[TrackId, _KFState],
        gmc: Dict[int, Transform],
    ) -> List[Tuple[Track, Detection]]:
        """Build (track, det) proposals for this frame."""
        results: List[Tuple[Track, Detection]] = []
        other_boxes = self._end_boxes(active_tracks)

        for det in detections:
            clean: List[Track] = []
            maybe: List[Track] = []

            for tr in active_tracks:
                decision = self._classify_pair(tr, det, other_boxes, kf_state, gmc)
                if decision == _ComparisonResult.MATCH:
                    clean.append(tr)
                elif decision == _ComparisonResult.MAY_MATCH:
                    maybe.append(tr)

            picked = self._pick_track_for_detection(clean, maybe)
            if picked is None:
                # will be turned into a brand-new track later
                tmp = Track(track_id=-1, sorted_detections=[det])  # temporary id
                results.append((tmp, det))
            else:
                results.append((picked, det))

        return results

    def _apply_matches(
        self,
        matches: List[Tuple[Track, Detection]],
        active_tracks: List[Track],
        kf_state: Dict[TrackId, _KFState],
        next_tid: int,
        gmc: Dict[int, Transform],
    ) -> Tuple[List[Track], List[Track], int]:
        """
        Resolve multiple detections per track, update KF+EMA, spawn new tracks.
        Returns: (new_active_tracks, newly_staled_tracks, next_tid)
        """
        per_tid: DefaultDict[int, List[Detection]] = defaultdict(list)
        tid_to_track: Dict[int, Track] = {}
        for tr, d in matches:
            per_tid[tr.track_id].append(d)
            tid_to_track[tr.track_id] = tr

        new_active: List[Track] = [t for t in active_tracks]  # will edit in place
        newly_staled: List[Track] = []

        for tid, dets in per_tid.items():
            if tid == -1:
                # spawn one new track per such detection
                for d in dets:
                    new_track = Track(track_id=next_tid, sorted_detections=[d])
                    new_active.append(new_track)
                    self._init_kf(kf_state, new_track)
                    next_tid += 1
                continue

            tr = next(t for t in new_active if t.track_id == tid)

            if len(dets) > 1:
                # ambiguous → stale current and split into new tracks
                newly_staled.append(tr)
                new_active = [t for t in new_active if t.track_id != tid]
                for d in dets:
                    new_track = Track(track_id=next_tid, sorted_detections=[d])
                    new_active.append(new_track)
                    self._init_kf(kf_state, new_track)
                    next_tid += 1
            else:
                # single extension
                d = dets[0]
                assert d.frame_idx not in (tr.detections_by_frame.keys()), (
                    f'Detection {d.frame_idx} already in track {tr.track_id}'
                )
                tr.sorted_detections.append(d)
                self._kf_update_to_det(kf_state[tr.track_id], d, gmc)
                self._update_track_ema(tr)

        return new_active, newly_staled, next_tid

    # ───────────────────────────── scoring ───────────────────────────── #

    def _classify_pair(
        self,
        track: Track,
        det: Detection,
        other_boxes: Dict[TrackId, BoundingBox],
        kf_state: Dict[TrackId, _KFState],
        gmc: Dict[int, Transform],
    ) -> _ComparisonResult:
        """Score motion (Mahalanobis d²) and appearance (χ²); apply gates + isolation."""
        # motion
        st = kf_state.get(track.track_id)
        if st is None:
            return _ComparisonResult.MAY_MATCH  # brand-new active track, be lenient

        m_pred, P_pred, _ = self._predict_to(det.frame_idx, st, gmc)
        d2 = float(
            self._kf.gating_distance(m_pred, P_pred, det.bbox.center_wh[None, :], only_position=True, metric='maha')[0]
        )

        # appearance (EMA is stored in track.sorted_detections[0].embedding)
        ema = track.sorted_detections[0].embedding
        chi2 = chi2_dist(ema, det.embedding)

        if d2 <= self.gate_strict_d2 and chi2 <= self.chi2_strict:
            return _ComparisonResult.MATCH

        if d2 <= self.gate_loose_d2 and chi2 <= self.chi2_loose and self._is_isolated(det, track.track_id, other_boxes):
            return _ComparisonResult.MAY_MATCH

        return _ComparisonResult.NO_MATCH

    # ───────────────────────────── primitives ───────────────────────────── #

    @staticmethod
    def _group_single_detections_by_frame(tracks: Iterable[Track]) -> Dict[FrameIndex, List[Detection]]:
        out: DefaultDict[FrameIndex, List[Detection]] = defaultdict(list)
        for t in tracks:
            assert len(t.sorted_detections) == 1, 'Greedy preprocessor expects single-detection tracks.'
            out[t.sorted_detections[0].frame_idx].append(t.sorted_detections[0])
        return out

    @staticmethod
    def _end_boxes(active: Iterable[Track]) -> Dict[TrackId, BoundingBox]:
        """
        Map tid -> bbox of the last detection's bbox.
        Used by the isolation heuristic.
        """
        return {tr.track_id: tr.end.bbox for tr in active}

    def _pick_track_for_detection(self, clean: List[Track], maybe: List[Track]) -> Track | None:
        """Tie-breaker for multiple candidates: prefer single clean, else single maybe, else None."""
        if len(clean) == 1:
            return clean[0]
        if len(clean) == 0 and len(maybe) == 1:
            return maybe[0]
        return None

    def _init_kf(self, kf_state: Dict[TrackId, _KFState], tr: Track) -> None:
        m, P = self._kf.initiate(tr.sorted_detections[0].bbox.center_wh)
        kf_state[tr.track_id] = _KFState(mean=m, cov=P, last_frame=tr.start_frame)

    def _predict_to(self, to_frame: int, st: _KFState, gmc: Dict[int, Transform]) -> Tuple[np.ndarray, np.ndarray, int]:
        if to_frame <= st.last_frame:
            return st.mean, st.cov, st.last_frame
        m, P = self._kf.advance_state_to_frame(st.mean, st.cov, gmc, st.last_frame, int(to_frame))
        return m, P, int(to_frame)

    def _kf_update_to_det(self, st: _KFState, det: Detection, transforms: Dict[int, Transform]) -> None:
        m, P, _ = self._predict_to(det.frame_idx, st, transforms)
        m2, P2 = self._kf.update(m, P, det.bbox.center_wh)
        st.mean, st.cov, st.last_frame = m2, P2, det.frame_idx

    def _update_track_ema(self, tr: Track) -> None:
        """Recompute EMA (stored in the first detection) after appending a single new detection."""
        assert tr.sorted_detections, 'Track has no detections.'
        tr.sorted_detections[0].embedding = (
            self.ema_alpha * tr.sorted_detections[0].embedding
            + (1.0 - self.ema_alpha) * tr.sorted_detections[-1].embedding
        )

    def _is_isolated(
        self,
        det: Detection,
        this_tid: TrackId,
        other_boxes: Dict[TrackId, BoundingBox],
    ) -> bool:
        """True if detection is far from and non-overlapping with other tracks."""
        for tid, b in other_boxes.items():
            if tid == this_tid:
                continue
            if det.bbox.iou(b) > self.iso_iou_max:
                return False
            if det.bbox.center.distance_to(b.center) <= self.iso_center_mul * det.bbox.width:
                return False
        return True
