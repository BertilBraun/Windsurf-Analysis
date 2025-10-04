from __future__ import annotations

import logging
from enum import Enum
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

from server.inference.bot_sort.cmc import CMC
from server.inference.src.util.similarity_helpers import Embedding, HistogramEmbedding
from server.inference.src.visualization.stabilize import Transform
from ...util.video_io import VideoInfo, VideoReader
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
        # Debugging/visualization
        debug_video_path: Optional[str] = None,
    ):
        self.gate_strict_d2 = float(gate_strict_d2)
        self.gate_loose_d2 = float(gate_loose_d2)
        self.chi2_strict = float(chi2_strict)
        self.chi2_loose = float(chi2_loose)
        self.ema_alpha = float(ema_alpha)
        self.max_frame_distance = int(max_frame_distance)
        self.debug_vis = debug_video_path is not None
        self.debug_video_path = debug_video_path
        # Debug trail history of camera translations (dx,dy) similar to BoT-SORT
        self._camera_translation_history = deque([], maxlen=30)

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

        # Preload frames for debugging if requested
        frame_dict: Dict[int, np.ndarray] = {}
        if self.debug_vis and self.debug_video_path:
            try:
                with VideoReader(self.debug_video_path) as reader:
                    for f_idx, frame in reader.read_frames():
                        frame_dict[int(f_idx)] = frame
            except Exception:
                frame_dict = {}

        def create_new_track(detection: Detection) -> Track:
            nonlocal next_track_id
            new_track = Track(track_id=next_track_id, sorted_detections=[detection])
            next_track_id += 1
            self._kf[new_track.track_id] = KFState.init(new_track)
            self._ema[new_track.track_id] = detection.embedding
            return new_track

        def stale_track(track: Track) -> None:
            nonlocal stale_track_ids, stale_tracks
            if track.track_id not in stale_track_ids:
                stale_track_ids.add(track.track_id)
                stale_tracks.append(track)
                self._kf.pop(track.track_id, None)
                self._ema.pop(track.track_id, None)

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
                    new_track = create_new_track(detection)
                    matches_this_frame.append((new_track, detection))
                    print(
                        f'New track candidate: {new_track.track_id} on frame {frame_idx} len(clean_matches): {len(clean_matches)} len(mby_matches): {len(mby_matches)}'
                    )
                    print(detection)
                    print(
                        f'Index of detection in detections_by_frame: {detections_by_frame[frame_idx].index(detection)}'
                    )
                    for track in active_tracks:
                        print(
                            f'Track: {track.track_id} result: {self._compare_detection_to_track(track, detection, cmc=cmc)}'
                        )
                    print('-' * 100)

                    # All tracks that "almost matched" become stale
                    for track in clean_matches + mby_matches:
                        # TODO this breaks because the active tracks are not updated and previous matches might continue to match...
                        stale_track(track)

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
                    stale_track(track)
                    for det in detections:
                        new_track = create_new_track(det)
                        print(f'Conflicting attachments → stale original, spawn new tracks: {new_track.track_id}')
                        active_tracks.append(new_track)
                else:
                    # single extension
                    det = detections[0]
                    track.sorted_detections.append(det)
                    if track not in active_tracks:  # NOTE: dirty hack to activate new tracks
                        active_tracks.append(track)
                    # update KF + EMA
                    self._kf[track.track_id] = self._kf[track.track_id].update_to_det(det, cmc)
                    self._ema[track.track_id] = self._ema[track.track_id].interpolate(det.embedding, self.ema_alpha)

            # ── age out stale tracks (too far behind current frame) ──
            for track in list(active_tracks):
                if track.end.frame_idx + self.max_frame_distance < frame_idx:
                    stale_track(track)

            # keep only non-stale active tracks
            active_tracks = [track for track in active_tracks if track.track_id not in stale_track_ids]

            # ── optional debug: visualize current frame before resolving conflicts ──
            if self.debug_vis:
                try:
                    import cv2  # type: ignore

                    # Build matrices for distances between active tracks and current detections
                    frame_dets: List[Detection] = detections_by_frame[frame_idx]
                    sorted_active = sorted(active_tracks, key=lambda t: t.track_id)
                    R = len(sorted_active)
                    C = len(frame_dets)
                    if R > 0 and C > 0:
                        d2_mat = np.full((R, C), np.nan, dtype=np.float32)
                        chi2_mat = np.full((R, C), np.nan, dtype=np.float32)
                        for r, tr in enumerate(sorted_active):
                            # Ensure KF/EMA state exists
                            st = self._kf.get(tr.track_id)
                            if st is None:
                                st = self._kf[tr.track_id] = KFState.init(tr)
                            pred = st.predict_to(frame_idx, cmc)
                            ema = self._ema.get(tr.track_id, tr.start.embedding)
                            for c, det in enumerate(frame_dets):
                                try:
                                    d2_mat[r, c] = float(pred.gating_distance(det.bbox.center_wh, only_position=True))
                                except Exception:
                                    d2_mat[r, c] = np.nan
                                try:
                                    assert isinstance(ema, HistogramEmbedding)
                                    chi2_mat[r, c] = float(ema.distance(det.embedding))
                                except Exception:
                                    chi2_mat[r, c] = np.nan

                        # Draw overlays on the current frame if available, else on a blank canvas
                        if frame_dict:
                            base = frame_dict.get(int(frame_idx))
                            if base is not None:
                                to_display = base.copy()
                            else:
                                to_display = np.full(
                                    (int(video_properties.height), int(video_properties.width), 3), 20, dtype=np.uint8
                                )
                        else:
                            to_display = np.full(
                                (int(video_properties.height), int(video_properties.width), 3), 20, dtype=np.uint8
                            )

                        # Draw camera motion trail
                        try:
                            import cv2  # type: ignore

                            # Append current frame's camera translation if available
                            t_curr = cmc._transforms_dict[frame_idx]
                            if t_curr is not None:
                                self._camera_translation_history.append(
                                    np.array([float(t_curr.dx), float(t_curr.dy)], dtype=float)
                                )

                            if len(self._camera_translation_history) > 0:
                                img_h, img_w = to_display.shape[:2]
                                center = (img_w // 2, img_h // 2)
                                pts = [center]
                                for dt in self._camera_translation_history:
                                    last_pt = pts[-1]
                                    nxt = (int(round(last_pt[0] + dt[0])), int(round(last_pt[1] + dt[1])))
                                    pts.append(nxt)

                                n = len(pts)
                                if n > 1:
                                    idx = (np.linspace(0, 255, n - 1)).astype(np.uint8)
                                    cmap = cv2.applyColorMap(idx, cv2.COLORMAP_AUTUMN)
                                    colors = [tuple(int(c) for c in cmap[i, 0, ::-1]) for i in range(n - 1)]
                                else:
                                    colors = [(0, 255, 255)]

                                for i in range(1, n):
                                    cv2.line(to_display, pts[i - 1], pts[i], colors[i - 1], 2)
                                # Mark most recent point
                                cv2.circle(to_display, pts[-1], 3, (0, 0, 0), -1)
                                cv2.circle(to_display, pts[-1], 2, (0, 255, 255), -1)
                        except Exception:
                            pass

                        # Draw detections
                        for i, det in enumerate(frame_dets):
                            bb = det.bbox
                            cv2.rectangle(
                                to_display, (int(bb.x1), int(bb.y1)), (int(bb.x2), int(bb.y2)), (255, 255, 255), 2
                            )
                            cv2.putText(
                                to_display,
                                f'Det {i}',
                                (int(bb.x2), max(0, int(bb.y2) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                        # Draw KF predictions for active tracks at this frame
                        for tr in sorted_active:
                            try:
                                st = self._kf.get(tr.track_id)
                                if st is None:
                                    st = self._kf[tr.track_id] = KFState.init(tr)
                                pred = st.predict_to(frame_idx, cmc)
                                cx, cy, w, h = pred.display_bbox(alpha=0.0)
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
                                    (255, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )
                            except Exception:
                                pass

                        # Heatmaps (style similar to BoT-SORT)
                        def build_heatmap(mat: np.ndarray, title: str, row_labels, col_labels, vmin=None, vmax=None):
                            if mat is None or mat.size == 0:
                                canvas = np.full((120, 240, 3), 30, dtype=np.uint8)
                                cv2.putText(
                                    canvas,
                                    f'{title}: no data',
                                    (8, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    (200, 200, 200),
                                    1,
                                    cv2.LINE_AA,
                                )
                                return canvas
                            m = mat.astype(np.float32)
                            if vmin is None or vmax is None:
                                m_min = float(np.nanmin(m)) if np.isfinite(m).any() else 0.0
                                m_max = float(np.nanmax(m)) if np.isfinite(m).any() else 1.0
                            else:
                                m_min = float(vmin)
                                m_max = float(vmax)
                            m = np.nan_to_num(m, nan=m_max)
                            m_clipped = np.clip(m, m_min, m_max)
                            denom = (m_max - m_min) if (m_max - m_min) >= 1e-6 else 1.0
                            norm = ((m_clipped - m_min) / denom * 255.0).astype(np.uint8)
                            heat = cv2.applyColorMap(norm, cv2.COLORMAP_AUTUMN)
                            cell_h, cell_w = 80, 80
                            hm = cv2.resize(
                                heat, (heat.shape[1] * cell_w, heat.shape[0] * cell_h), interpolation=cv2.INTER_AREA
                            )
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
                            cv2.putText(
                                canvas, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA
                            )
                            for r, lbl in enumerate(row_labels):
                                y = top_margin + r * cell_h + int(cell_h * 0.7)
                                cv2.putText(
                                    canvas,
                                    f'Track {str(lbl)}',
                                    (4, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (200, 200, 200),
                                    1,
                                    cv2.LINE_AA,
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
                            # Annotate each cell with its numeric value
                            try:
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                rows, cols = m.shape
                                for r in range(rows):
                                    for c in range(cols):
                                        val = float(mat[r, c])
                                        label = f'{val:.2f}'
                                        (tw, th), _ = cv2.getTextSize(label, font, 0.35, 1)
                                        cx = left_margin + c * cell_w + cell_w // 2
                                        cy = top_margin + r * cell_h + cell_h // 2 + th // 2
                                        # Outline for readability
                                        cv2.putText(
                                            canvas, label, (cx - tw // 2, cy), font, 0.35, (0, 0, 0), 2, cv2.LINE_AA
                                        )
                                        cv2.putText(
                                            canvas,
                                            label,
                                            (cx - tw // 2, cy),
                                            font,
                                            0.35,
                                            (255, 255, 255),
                                            1,
                                            cv2.LINE_AA,
                                        )
                            except Exception:
                                pass
                            return canvas

                        row_labels = [t.track_id for t in sorted_active]
                        col_labels = list(range(C))
                        hm_chi2 = build_heatmap(chi2_mat, 'chi2 embedding (low=better)', row_labels, col_labels)
                        hm_d2 = build_heatmap(
                            d2_mat, 'KF gating g2 (pos-only)', row_labels, col_labels, vmin=0.0, vmax=20.0
                        )
                        heat_row = np.concatenate([hm_chi2, hm_d2], axis=1)

                        cv2.imshow('greedy_frame', to_display)
                        cv2.imshow('greedy_heatmaps (lower = better)', heat_row)
                        cv2.waitKey(0)
                except Exception:
                    pass

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

        # Predict to current frame (cached inside KFState)
        pred = self._kf[tid].predict_to(detection.frame_idx, cmc)

        # Motion gating: squared Mahalanobis distance (df=2)
        d2 = pred.gating_distance(detection.bbox.center_wh)

        # Appearance: χ² on L1-normalized histograms with EMA
        ema = self._ema[tid]
        assert isinstance(ema, HistogramEmbedding)
        chi2 = ema.distance(detection.embedding)

        print(f'd2: {d2} chi2: {chi2} for track {tid} and detection {detection.frame_idx} bbox: {detection.bbox}')

        # Decision
        if d2 <= self.gate_strict_d2 and chi2 <= self.chi2_strict:
            return _ComparisonResult.MATCH

        if d2 <= self.gate_loose_d2 and chi2 <= self.chi2_loose:
            # Isolation logic intentionally disabled for now.
            # If needed later, reintroduce a tie-break under the loose gate.
            return _ComparisonResult.MAY_MATCH

        return _ComparisonResult.NO_MATCH
