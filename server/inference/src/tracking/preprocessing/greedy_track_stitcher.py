from __future__ import annotations

import logging
from enum import Enum
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

from server.inference.bot_sort.cmc import CMC
from server.inference.src.util.algebra import probability_from_dist
from server.inference.src.util.similarity_helpers import Embedding
from server.inference.src.visualization.debug.session import DebugSession
from server.inference.src.visualization.stabilize import Transform
from ...util.video_io import VideoInfo
from ...common_types import Detection, FrameIndex, Track, TrackId
from server.inference.bot_sort.kalman_filter import KFState
from server.inference.src.visualization.debug import get_debug_session
from server.inference.src.visualization.debug.overlays import DetectionsOverlay, KalmanOverlay
from server.inference.src.visualization.debug.draw import draw_heatmap


class _ComparisonResult(Enum):
    MATCH = 'match'
    MAY_MATCH = 'may_match'
    NO_MATCH = 'no_match'


# ───────────────────────────── main class ───────────────────────────── #


class GreedyTrackStitcher:
    """
    Greedy pre-stitcher with original matching orchestration, but using:

      • Motion: KF + forward GMC deltas (prev→curr) → position-only Mahalanobis d²
        - Values 0-0.2 are very good matches
        - Values 0.2-5.0 are still good matches
        - Values >5.0 (up to tens of thousands) are bad matches

      • Appearance: χ² distance on L1-normalized L*a*b* histograms (with EMA)
        - Values 0-0.05 are very good matches
        - Values 0.05-0.15 are still good matches
        - Values >0.15 (up to ~0.9 for the worst matching pairs in the sample dataset) are bad matches

    Decision:
      MATCH      if (d² ≤ motion_strict) and (χ² ≤ appearance_strict)
      MAY_MATCH  if (d² ≤ motion_loose)  and (χ² ≤ appearance_loose)
      NO_MATCH   otherwise
    """

    def __init__(
        self,
        *,
        motion_probability_strict: float,
        motion_probability_loose: float,
        appearance_probability_strict: float,
        appearance_probability_loose: float,
        ema_alpha: float,
        max_frame_distance: int,
        # Debugging/visualization
        debug_video_path: Optional[str] = None,
    ):
        self.motion_probability_strict = float(motion_probability_strict)
        self.motion_probability_loose = float(motion_probability_loose)
        self.appearance_probability_strict = float(appearance_probability_strict)
        self.appearance_probability_loose = float(appearance_probability_loose)
        self.ema_alpha = float(ema_alpha)
        self.max_frame_distance = int(max_frame_distance)

        # TODO remove
        self.debug_vis = debug_video_path is not None
        self.debug_video_path = debug_video_path
        # Debug trail history of camera translations (dx,dy) similar to BoT-SORT
        self._camera_translation_history = deque([], maxlen=30)

    # ─────────────────────────── public API ─────────────────────────── #

    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        """Greedily stitches single-detection tracks into longer tracks (old orchestration, new compare)."""
        logging.info(f'{"=" * 30} Running greedy preprocessor with {len(tracks)} tracks {"=" * 30}')

        # Build per-frame detections from single-detection inputs
        detections_by_frame: Dict[FrameIndex, List[Detection]] = defaultdict(list)
        for track in tracks:
            assert len(track.sorted_detections) == 1, 'Greedy preprocessor only supports single-detection tracks'
            for detection in track.sorted_detections:
                detections_by_frame[detection.frame_idx].append(detection)

        frames: List[int] = sorted(int(f) for f in detections_by_frame.keys())

        cmc = CMC(transforms)

        next_track_id: int = 1
        # Per-track state
        kf: Dict[TrackId, KFState] = {}
        ema: Dict[TrackId, Embedding] = {}

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
            kf[new_track.track_id] = KFState.init(new_track.sorted_detections[0])
            ema[new_track.track_id] = detection.embedding
            return new_track

        def stale_track(track: Track) -> None:
            nonlocal stale_track_ids, stale_tracks
            if track.track_id not in stale_track_ids:
                stale_track_ids.add(track.track_id)
                stale_tracks.append(track)
                kf.pop(track.track_id, None)
                ema.pop(track.track_id, None)

        def update_track(track: Track, detection: Detection, cmc: CMC) -> None:
            track.sorted_detections.append(detection)
            kf[track.track_id] = kf[track.track_id].update_to_det(detection, cmc)
            ema[track.track_id] = ema[track.track_id].interpolate(detection.embedding, self.ema_alpha)

        for frame_idx in frames:
            frame_detections: List[Detection] = detections_by_frame[frame_idx]
            track_by_id: Dict[int, Track] = {t.track_id: t for t in active_tracks}

            valid_proposals, detections_to_create_new_tracks, fade_track_ids = self._generate_tentative_proposals(
                frame_detections, active_tracks, cmc, kf, ema
            )

            # Stale faded tracks now (single commit point)
            for track_id in fade_track_ids:
                stale_track(track_by_id[track_id])

            # Apply surviving 1:1 proposals
            for track_id, detection in valid_proposals:
                assert track_id in track_by_id, f'Track {track_id} not found in track_by_id but must be present'
                track = track_by_id[track_id]
                if track.track_id in stale_track_ids:
                    # Track no longer active (e.g., faded); convert to new track
                    detections_to_create_new_tracks.add(detection)
                else:
                    update_track(track, detection, cmc)

            # ── age out stale tracks (too far behind current frame) ──
            for track in active_tracks:
                if track.end.frame_idx + self.max_frame_distance < frame_idx:
                    stale_track(track)

            # keep only non-stale active tracks
            active_tracks = [track for track in active_tracks if track.track_id not in stale_track_ids]

            # Create new tracks for remaining detections
            for detection in detections_to_create_new_tracks:
                active_tracks.append(create_new_track(detection))

            # ── optional debug: visualize current frame before resolving conflicts ──
            if self.debug_vis:
                try:
                    import cv2  # type: ignore

                    # Build matrices for distances between active tracks and current detections
                    frame_detections_dbg: List[Detection] = detections_by_frame[frame_idx]
                    sorted_active_tracks = sorted(active_tracks, key=lambda t: t.track_id)
                    num_rows = len(sorted_active_tracks)
                    num_cols = len(frame_detections_dbg)
                    if num_rows > 0 and num_cols > 0:
                        d2_matrix = np.full((num_rows, num_cols), np.nan, dtype=np.float32)
                        chi2_matrix = np.full((num_rows, num_cols), np.nan, dtype=np.float32)
                        for row_index, track in enumerate(sorted_active_tracks):
                            # Ensure KF/EMA state exists
                            kf_state = kf.get(track.track_id)
                            if kf_state is None:
                                kf_state = kf[track.track_id] = KFState.init(track.sorted_detections[0])
                            pred = kf_state.predict_to(frame_idx, cmc)
                            track_ema = ema.get(track.track_id, track.start.embedding)
                            for col_index, detection in enumerate(frame_detections_dbg):
                                try:
                                    d2_matrix[row_index, col_index] = float(
                                        pred.gating_distance(detection.bbox.center_wh, only_position=True)
                                    )
                                except Exception:
                                    d2_matrix[row_index, col_index] = np.nan
                                try:
                                    assert isinstance(track_ema, HistogramEmbedding)
                                    chi2_matrix[row_index, col_index] = float(track_ema.distance(detection.embedding))
                                except Exception:
                                    chi2_matrix[row_index, col_index] = np.nan

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
                        for i, detection in enumerate(frame_detections_dbg):
                            bbox = detection.bbox
                            cv2.rectangle(
                                to_display,
                                (int(bbox.x1), int(bbox.y1)),
                                (int(bbox.x2), int(bbox.y2)),
                                (255, 255, 255),
                                2,
                            )
                            cv2.putText(
                                to_display,
                                f'Det {i}',
                                (int(bbox.x2), max(0, int(bbox.y2) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )

                        # Draw KF predictions for active tracks at this frame
                        for track in sorted_active_tracks:
                            try:
                                kf_state = kf.get(track.track_id)
                                if kf_state is None:
                                    kf_state = kf[track.track_id] = KFState.init(track.sorted_detections[0])
                                pred = kf_state.predict_to(frame_idx, cmc)
                                cx, cy, w, h = pred.display_bbox(alpha=0.0)
                                x1 = int(cx - w / 2.0)
                                y1 = int(cy - h / 2.0)
                                x2 = int(cx + w / 2.0)
                                y2 = int(cy + h / 2.0)
                                cv2.rectangle(to_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(
                                    to_display,
                                    f'KF id={track.track_id}',
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

                        row_labels = [t.track_id for t in sorted_active_tracks]
                        col_labels = list(range(num_cols))
                        hm_chi2 = build_heatmap(chi2_matrix, 'chi2 embedding (low=better)', row_labels, col_labels)
                        hm_d2 = build_heatmap(
                            d2_matrix, 'KF gating g2 (pos-only)', row_labels, col_labels, vmin=0.0, vmax=20.0
                        )
                        heat_row = np.concatenate([hm_chi2, hm_d2], axis=1)

                        cv2.imshow('greedy_frame', to_display)
                        cv2.imshow('greedy_heatmaps (lower = better)', heat_row)
                        cv2.waitKey(0)
                except Exception:
                    pass

        return stale_tracks + active_tracks

    # ───────────────────────────── scoring ───────────────────────────── #

    def _generate_tentative_proposals(
        self,
        frame_detections: List[Detection],
        active_tracks: List[Track],
        cmc: CMC,
        kf: Dict[TrackId, KFState],
        ema: Dict[TrackId, Embedding],
    ) -> tuple[List[tuple[int, Detection]], set[Detection], set[int]]:
        # Tentative proposals and fade/new collections
        tentative_proposals: List[tuple[int, Detection]] = []
        detections_to_create_new_tracks: set[Detection] = set()
        fade_track_ids: set[int] = set()

        for detection in frame_detections:
            clean_candidates: List[Track] = []
            maybe_candidates: List[Track] = []

            for candidate_track in active_tracks:
                result = self._compare_detection_to_track(
                    candidate_track,
                    detection,
                    cmc=cmc,
                    kf=kf[candidate_track.track_id],
                    ema=ema[candidate_track.track_id],
                )
                if result == _ComparisonResult.MATCH:
                    clean_candidates.append(candidate_track)
                elif result == _ComparisonResult.MAY_MATCH:
                    maybe_candidates.append(candidate_track)

            # Clean-first rule with uniqueness
            if len(clean_candidates) == 1:
                tentative_proposals.append((clean_candidates[0].track_id, detection))
                for track in maybe_candidates:  # All maybe candidates are faded - remove them
                    fade_track_ids.add(track.track_id)
            elif len(clean_candidates) == 0 and len(maybe_candidates) == 1:
                tentative_proposals.append((maybe_candidates[0].track_id, detection))
            else:
                # Ambiguous or no candidates → new track; fade all candidates for this detection
                detections_to_create_new_tracks.add(detection)
                for track in clean_candidates + maybe_candidates:
                    fade_track_ids.add(track.track_id)

        # Commit fades: any track with multiple proposals in this frame must be faded
        duplicates = self._get_duplicated_assignment_proposals(tentative_proposals)
        for track_id, detection_list in duplicates:
            fade_track_ids.add(track_id)
            detections_to_create_new_tracks.update(detection_list)

        # Finalize proposals by removing any that reference faded tracks
        finalized_proposals = self._get_non_stale_proposals(tentative_proposals, fade_track_ids)

        # Any detection that lost its proposal due to fading becomes a new track
        for track_id, det in tentative_proposals:
            if track_id in fade_track_ids:
                detections_to_create_new_tracks.add(det)

        return finalized_proposals, detections_to_create_new_tracks, fade_track_ids

    def _compare_detection_to_track(
        self,
        track: Track,
        detection: Detection,
        cmc: CMC,
        kf: KFState,
        ema: Embedding,
    ) -> _ComparisonResult:
        """
        Score motion (Mahalanobis d² on [cx,cy]) + appearance (χ² on L1-hist EMA).
        """
        # Early guard on excessive frame gap (keeps candidate set small)
        gap = detection.frame_idx - track.end.frame_idx
        if gap <= 0 or gap > self.max_frame_distance:
            return _ComparisonResult.NO_MATCH

        # Predict to current frame (cached inside KFState)
        pred = kf.predict_to(detection.frame_idx, cmc)

        # Motion gating: squared Mahalanobis distance (df=2)
        d2 = pred.gating_distance(detection.bbox.center_wh)
        motion_probability = probability_from_dist(d2, df=2)

        # Appearance: χ² on L1-normalized histograms with EMA
        appearance_probability = ema.probability(detection.embedding)

        # Decision
        if (
            motion_probability >= self.motion_probability_strict
            and appearance_probability >= self.appearance_probability_strict
            and gap <= 3
        ):
            # Strong matches are only allowed on frame gaps of 3 or less
            return _ComparisonResult.MATCH

        if (
            motion_probability >= self.motion_probability_strict
            or appearance_probability >= self.appearance_probability_strict
        ):
            return _ComparisonResult.MAY_MATCH

        if (
            motion_probability >= self.motion_probability_loose
            and appearance_probability >= self.appearance_probability_loose
        ):
            # Isolation logic intentionally disabled for now.
            # If needed later, reintroduce a tie-break under the loose gate.
            return _ComparisonResult.MAY_MATCH

        return _ComparisonResult.NO_MATCH

    @staticmethod
    def _get_non_stale_proposals(
        proposals: List[tuple[int, Detection]], fade_track_ids: set[int]
    ) -> List[tuple[int, Detection]]:
        return [(track_id, detection) for (track_id, detection) in proposals if track_id not in fade_track_ids]

    @staticmethod
    def _get_duplicated_assignment_proposals(
        proposals: List[tuple[int, Detection]],
    ) -> List[tuple[int, List[Detection]]]:
        proposals_by_track: Dict[int, List[Detection]] = defaultdict(list)
        for track_id, detection in proposals:
            proposals_by_track[track_id].append(detection)

        return [
            (track_id, detection_list)
            for (track_id, detection_list) in proposals_by_track.items()
            if len(detection_list) > 1
        ]
