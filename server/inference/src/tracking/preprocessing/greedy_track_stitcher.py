from __future__ import annotations

from bisect import bisect_right
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import DefaultDict, Dict, Iterable, List, Tuple, Optional
from collections import defaultdict

import numpy as np

from server.inference.src.settings import EPS
from server.inference.src.visualization.stabilize import Transform
from ...util.video_io import VideoInfo, VideoReader
from ...common_types import BoundingBox, Detection, FrameIndex, Track, TrackId
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


KF = KalmanFilter()


@dataclass(frozen=True)
class _KFState:
    mean: np.ndarray  # (8,)
    cov: np.ndarray  # (8,8)
    last_frame: int

    _cached_predictions: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = field(default_factory=dict)

    def predict_to(self, to_frame: int, gmc: Dict[int, Transform]) -> Tuple[np.ndarray, np.ndarray, int]:
        if to_frame <= self.last_frame:
            return self.mean, self.cov, self.last_frame
        if to_frame in self._cached_predictions:
            return self._cached_predictions[to_frame]

        start_m, start_P, start_frame = self.mean, self.cov, self.last_frame

        frames = sorted(self._cached_predictions.keys())
        idx = bisect_right(frames, to_frame - 1) - 1
        if idx >= 0 and frames[idx] <= to_frame and frames[idx] > start_frame:
            start_m, start_P, start_frame = self._cached_predictions[frames[idx]]

        m, P = KF.advance_state_to_frame(start_m, start_P, gmc, start_frame, int(to_frame))
        self._cached_predictions[to_frame] = (m, P, int(to_frame))
        return m, P, int(to_frame)

    def update_to_det(self, det: Detection, gmc: Dict[int, Transform]) -> _KFState:
        m, P, _ = self.predict_to(det.frame_idx, gmc)
        m2, P2 = KF.update(m, P, det.bbox.center_wh)
        return _KFState(mean=m2, cov=P2, last_frame=det.frame_idx)

    @staticmethod
    def init_kf(tr: Track) -> _KFState:
        m, P = KF.initiate(tr.sorted_detections[0].bbox.center_wh)
        return _KFState(mean=m, cov=P, last_frame=tr.start_frame)


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
        gate_strict_d2: float,
        gate_loose_d2: float,  # χ²(2, 0.95)
        chi2_strict: float,  # tune on data
        chi2_loose: float,  # tune on data
        iso_center_mul: float,  # isolation: center dist > iso_center_mul * det.width
        iso_iou_max: float,  # and IoU to others <= iso_iou_max
        max_frame_distance: int = 5,  # stale cutoff (frames)
        ema_alpha: float = 0.9,  # appearance EMA smoothing
        # TODO: Debug options
        debug_video_path: Optional[str] = None,
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

        # Debug config/state
        self.debug_vis = False  # True
        self.debug_wait_ms = 0  # TODO remove
        self.debug_video_path = debug_video_path
        self._debug_frames: Optional[Dict[int, np.ndarray]] = None

    # ─────────────────────────── public API ─────────────────────────── #

    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        """Greedily stitches single-detection tracks into longer tracks."""
        logging.info(f'{"=" * 28} Greedy preprocessor: {len(tracks)} inputs {"=" * 28}')
        dets_by_frame = self._group_single_detections_by_frame(tracks)
        frames = sorted(dets_by_frame.keys())
        gmc = {int(t.frame_idx): t for t in transforms}  # per-frame forward delta (f-1 → f)

        # Lazy-load frames for debug if requested
        if self.debug_vis and self._debug_frames is None and self.debug_video_path is not None:
            try:
                import cv2  # type: ignore

                _ = cv2  # silence linter unused
                self._debug_frames = {}
                with VideoReader(self.debug_video_path) as reader:
                    for f_idx, frame in reader.read_frames():
                        self._debug_frames[int(f_idx)] = frame
            except Exception:
                # If OpenCV is not available or video cannot be read, disable debug
                self._debug_frames = None
                self.debug_vis = False

        active: List[Track] = []
        stale: List[Track] = []
        stale_ids: set[TrackId] = set()
        kf_state: Dict[TrackId, _KFState] = {}
        next_tid = 1

        for f in frames:
            candidates = dets_by_frame[f]
            matches, picked_by_det, d2_mat, chi2_mat = self._collect_matches_for_frame(
                f, candidates, active, kf_state, gmc
            )

            # Optional debug visualization for this frame
            if self.debug_vis:
                try:
                    self._debug_show_frame_and_heatmaps(
                        frame=f,
                        detections=candidates,
                        active_tracks=active,
                        kf_state=kf_state,
                        gmc=gmc,
                        picked_by_det=picked_by_det,
                        d2_mat=d2_mat,
                        chi2_mat=chi2_mat,
                    )
                except Exception:
                    pass

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
    ) -> Tuple[
        List[Tuple[Track, Detection]],
        Dict[int, Optional[Track]],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """Build (track, det) proposals for this frame. Returns matches and debug data."""
        results: List[Tuple[Track, Detection]] = []
        other_boxes = self._end_boxes(active_tracks)

        # Prepare matrices for debug heatmaps (rows: tracks, cols: detections)
        sorted_tracks = sorted(active_tracks, key=lambda t: t.track_id)
        d2_mat = np.full((len(sorted_tracks), len(detections)), np.nan, dtype=np.float32) if detections else None
        chi2_mat = np.full((len(sorted_tracks), len(detections)), np.nan, dtype=np.float32) if detections else None

        # Compute pairwise distances for debug
        if d2_mat is not None and chi2_mat is not None:
            for r, tr in enumerate(sorted_tracks):
                st = kf_state.get(tr.track_id)
                for c, det in enumerate(detections):
                    try:
                        if st is not None:
                            m_pred, P_pred, _ = st.predict_to(det.frame_idx, gmc)
                            d2 = float(
                                self._kf.gating_distance(
                                    m_pred, P_pred, det.bbox.center_wh[None, :], only_position=True, metric='maha'
                                )[0]
                            )
                            d2_mat[r, c] = d2
                        ema = tr.sorted_detections[0].embedding
                        chi2_mat[r, c] = chi2_dist(ema, det.embedding)
                    except Exception:
                        pass

        picked_by_det: Dict[int, Optional[Track]] = {}
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
            picked_by_det[det.frame_idx * 1000000 + id(det)] = picked  # unique key per det instance
            if picked is None:
                # will be turned into a brand-new track later
                tmp = Track(track_id=-1, sorted_detections=[det])  # temporary id
                results.append((tmp, det))  # TODO clean and maybe are now stale, right?
            else:
                results.append((picked, det))

        return results, picked_by_det, d2_mat, chi2_mat

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
                    kf_state[new_track.track_id] = _KFState.init_kf(new_track)
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
                    kf_state[new_track.track_id] = _KFState.init_kf(new_track)
                    next_tid += 1
            else:
                # single extension
                d = dets[0]
                assert d.frame_idx not in (tr.detections_by_frame.keys()), (
                    f'Detection {d.frame_idx} already in track {tr.track_id}'
                )
                tr.sorted_detections.append(d)
                kf_state[tr.track_id] = kf_state[tr.track_id].update_to_det(d, gmc)
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

        m_pred, P_pred, _ = st.predict_to(det.frame_idx, gmc)
        d2 = float(
            self._kf.gating_distance(m_pred, P_pred, det.bbox.center_wh[None, :], only_position=True, metric='maha')[0]
        )

        # appearance (EMA is stored in track.sorted_detections[0].embedding)
        ema = track.sorted_detections[0].embedding
        chi2 = chi2_dist(ema, det.embedding)

        if self.debug_vis:
            last_bbox = track.sorted_detections[-1].bbox
            print(
                f'Track {track.track_id} d2 motion: {d2}, chi2 embedding: {chi2}, track bbox: [{last_bbox.x1}, {last_bbox.y1}, {last_bbox.x2}, {last_bbox.y2}] detection bbox: [{det.bbox.x1}, {det.bbox.y1}, {det.bbox.x2}, {det.bbox.y2}]'
            )

        if d2 <= self.gate_strict_d2 and chi2 <= self.chi2_strict:
            return _ComparisonResult.MATCH

        if d2 <= self.gate_loose_d2 and chi2 <= self.chi2_loose:
            return _ComparisonResult.MAY_MATCH

        if self._is_isolated(det, track.track_id, other_boxes):
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
            if self.debug_vis:
                print(f'Picked clean track {clean[0].track_id}')
            return clean[0]
        if len(clean) == 0 and len(maybe) == 1:
            if self.debug_vis:
                print(f'Picked maybe track {maybe[0].track_id}')
            return maybe[0]
        if self.debug_vis:
            print(f'Picked no track: {len(clean)} clean, {len(maybe)} maybe')
        return None

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

    # ───────────────────────────── debug helpers ───────────────────────────── #

    def _debug_show_frame_and_heatmaps(
        self,
        *,
        frame: int,
        detections: List[Detection],
        active_tracks: List[Track],
        kf_state: Dict[TrackId, _KFState],
        gmc: Dict[int, Transform],
        picked_by_det: Dict[int, Optional[Track]],
        d2_mat: Optional[np.ndarray],
        chi2_mat: Optional[np.ndarray],
    ) -> None:
        if not self.debug_vis:
            return
        try:
            import cv2  # type: ignore
        except Exception:
            return

        # Draw current frame with detections and KF predictions
        frame_img = None
        if self._debug_frames is not None:
            frame_img = self._debug_frames.get(int(frame))
        if frame_img is None:
            return

        to_display = frame_img.copy()

        # Draw detections (white)
        for idx, d in enumerate(detections):
            bb = d.bbox
            cv2.rectangle(to_display, (int(bb.x1), int(bb.y1)), (int(bb.x2), int(bb.y2)), (255, 255, 255), 2)
            cv2.putText(
                to_display,
                f'Det {idx}',
                (int(bb.x2), max(0, int(bb.y2) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw KF predictions for active tracks (green)
        for tr in active_tracks:
            st = kf_state.get(tr.track_id)
            if st is None:
                continue
            try:
                m_pred, P_pred, _ = st.predict_to(frame, gmc)
                cx, cy, w, h = self._kf.display_bbox(m_pred, P_pred, alpha=0.0)
                x1 = int(cx - w / 2.0)
                y1 = int(cy - h / 2.0)
                x2 = int(cx + w / 2.0)
                y2 = int(cy + h / 2.0)
                cv2.rectangle(to_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    to_display,
                    f'KF id={tr.track_id}',
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

        # Draw association lines for picked pairs with g^2
        # Build lookup of detection indices for stable labeling
        for d in detections:
            picked = picked_by_det.get(d.frame_idx * 1000000 + id(d))
            if picked is None or picked.track_id == -1:
                continue
            try:
                st = kf_state.get(picked.track_id)
                if st is None:
                    continue
                m_pred, P_pred, _ = st.predict_to(d.frame_idx, gmc)
                z = d.bbox.center_wh.astype(np.float64).reshape(1, 4)
                g2 = float(self._kf.gating_distance(m_pred, P_pred, z, only_position=True, metric='maha')[0])

                # Centers
                cx, cy, w, h = self._kf.display_bbox(m_pred, P_pred, alpha=0.0)
                tcx, tcy = int(round(cx)), int(round(cy))
                dcx = int(round((d.bbox.x1 + d.bbox.x2) / 2.0))
                dcy = int(round((d.bbox.y1 + d.bbox.y2) / 2.0))
                cv2.line(to_display, (tcx, tcy), (dcx, dcy), (0, 200, 255), 1)
                mx = int((tcx + dcx) / 2)
                my = int((tcy + dcy) / 2)
                cv2.putText(
                    to_display,
                    f'g^2={g2:.2f}',
                    (mx + 4, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

        # Heatmaps for chi2 and gating distances
        def build_heatmap(
            mat: Optional[np.ndarray], title: str, row_labels, col_labels, vmin=None, vmax=None
        ) -> np.ndarray:
            h = 120
            w = 240
            if mat is None or mat.size == 0:
                canvas = np.full((h, w, 3), 30, dtype=np.uint8)
                cv2.putText(
                    canvas,
                    f'{title}: no data',
                    (8, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                return canvas
            m = mat.astype(np.float32)
            # handle NaNs
            m = np.where(np.isnan(m), 0.0, m)
            m_min = float(np.min(m)) if vmin is None else float(vmin)
            m_max = float(np.max(m)) if vmax is None else float(vmax)
            m_clipped = np.clip(m, m_min, m_max)
            denom = (m_max - m_min) if (m_max - m_min) >= 1e-6 else 1.0
            norm = ((m_clipped - m_min) / denom * 255.0).astype(np.uint8)
            heat = cv2.applyColorMap(norm, cv2.COLORMAP_AUTUMN)
            cell_h, cell_w = 80, 80
            hm = cv2.resize(heat, (heat.shape[1] * cell_w, heat.shape[0] * cell_h), interpolation=cv2.INTER_AREA)
            top_margin, left_margin = 50, 36
            colorbar_w, colorbar_gap = 16, 8
            canvas_h = hm.shape[0] + top_margin
            canvas_w = hm.shape[1] + left_margin + colorbar_gap + colorbar_w
            canvas = np.full((canvas_h, canvas_w, 3), 15, dtype=np.uint8)
            canvas[top_margin:, left_margin : left_margin + hm.shape[1]] = hm
            rows, cols = m.shape
            for r in range(rows + 1):
                y = top_margin + r * cell_h
                cv2.line(canvas, (left_margin, y), (left_margin + cols * cell_w, y), (60, 60, 60), 1)
            for c in range(cols + 1):
                x = left_margin + c * cell_w
                cv2.line(canvas, (x, top_margin), (x, top_margin + rows * cell_h), (60, 60, 60), 1)
            cv2.putText(canvas, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            for r, lbl in enumerate(row_labels):
                y = top_margin + r * cell_h + int(cell_h * 0.7)
                cv2.putText(
                    canvas, f'Track {str(lbl)}', (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA
                )
            for c, lbl in enumerate(col_labels):
                x = left_margin + c * cell_w + 2
                cv2.putText(
                    canvas,
                    f'Det {str(lbl)}',
                    (x, top_margin - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
            # Colorbar
            bar_h = hm.shape[0]
            grad = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
            grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_AUTUMN)
            x0 = left_margin + hm.shape[1] + colorbar_gap
            canvas[top_margin:, x0 : x0 + colorbar_w] = cv2.resize(
                grad_color, (colorbar_w, bar_h), interpolation=cv2.INTER_AREA
            )
            cv2.putText(
                canvas,
                'high',
                (x0 - 2, top_margin + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                'low',
                (x0 + 2, top_margin + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f'{m_max:.2f}',
                (x0 + colorbar_w + 4, top_margin + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f'{m_min:.2f}',
                (x0 + colorbar_w + 4, top_margin + bar_h - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            # Annotate cells with values
            font = cv2.FONT_HERSHEY_SIMPLEX
            for r in range(rows):
                for c in range(cols):
                    val = float(m[r, c])
                    label = f'{val:.2f}'
                    (tw, th), _ = cv2.getTextSize(label, font, 0.35, 1)
                    cx = left_margin + c * cell_w + cell_w // 2
                    cy = top_margin + r * cell_h + cell_h // 2 + th // 2
                    cv2.putText(canvas, label, (cx - tw // 2, cy), font, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(canvas, label, (cx - tw // 2, cy), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            return canvas

        row_labels = [t.track_id for t in sorted(active_tracks, key=lambda t: t.track_id)]
        col_labels = list(range(len(detections)))
        hm_chi2 = build_heatmap(chi2_mat, 'Chi2 distance', row_labels, col_labels)
        hm_d2 = build_heatmap(d2_mat, 'KF gating g^2 (0-20)', row_labels, col_labels, vmin=0.0, vmax=20.0)
        heat_row = np.concatenate([hm_chi2, hm_d2], axis=1)

        cv2.imshow('greedy_stitcher', to_display)
        cv2.imshow('greedy_heatmaps', heat_row)
        cv2.waitKey(self.debug_wait_ms)
