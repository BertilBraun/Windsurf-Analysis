from __future__ import annotations

import logging
from enum import Enum
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

from inference.src.motion.cmc import CMC
from inference.src.util.algebra import probability_from_dist
from inference.src.util.similarity_helpers import Embedding
from inference.src.visualization.debug.session import DebugSession
from inference.src.visualization.stabilize import Transform
from ...util.video_io import VideoInfo
from ...common_types import Detection, FrameIndex, Track, TrackId
from inference.src.motion.kalman_filter import KFState
from inference.src.visualization.debug import get_debug_session
from inference.src.visualization.debug.overlays import DetectionsOverlay, KalmanOverlay
from inference.src.visualization.debug.draw import draw_heatmap


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

        cmc = CMC(transforms)

        next_track_id: int = 1
        # Per-track state
        kf: Dict[TrackId, KFState] = {}
        ema: Dict[TrackId, Embedding] = {}

        # Active = can be extended; Stale = finalized and kept for output
        active_tracks: List[Track] = []
        stale_tracks: List[Track] = []
        stale_track_ids: set[TrackId] = set()

        # Initialize debug session if requested
        debug = get_debug_session(self.debug_video_path or '', enabled=self.debug_vis)

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

        def split_track_at_last_detection(track: Track) -> None:
            """
            Split an active track into:
              - a finalized prefix (up to frame before its last detection)
              - a new track seeded with its last detection

            Used to resolve overlapping detections: if a new track is started for a detection that overlaps
            another track's detection on the same frame, we end (stale) the other track and restart it from
            that last detection as a fresh track id.
            """
            # Only meaningful if there's something to keep as "prefix"
            if len(track.sorted_detections) < 2:
                return
            if track.track_id in stale_track_ids:
                return

            last_det = track.sorted_detections.pop()  # last detection becomes the seed for a new track
            # Finalize the prefix
            stale_track(track)
            # Ensure the stale track is removed from active list (we already filtered earlier in the loop)
            try:
                active_tracks.remove(track)
            except ValueError:
                pass
            # Restart from the last detection
            active_tracks.append(create_new_track(last_det))

        for frame_idx, frame_detections in detections_by_frame.items():
            track_by_id: Dict[int, Track] = {t.track_id: t for t in active_tracks}

            valid_proposals, detections_to_create_new_tracks, fade_track_ids = self._generate_tentative_proposals(
                frame_detections, active_tracks, cmc, kf, ema
            )

            # ── optional debug: visualize match scores before updates ──
            self._debug_visualize(detections_by_frame, frame_idx, active_tracks, kf, ema, cmc, debug, video_properties)

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

            # Create new tracks for remaining detections.
            # If a new track is started on this frame and it overlaps an existing track's detection on this frame,
            # split the existing track at this frame (end prefix, restart from its last detection).
            for detection in detections_to_create_new_tracks:
                overlapping_tracks = [
                    t
                    for t in list(active_tracks)
                    if t.end_frame == detection.frame_idx and t.end.bbox.overlaps(detection.bbox)
                ]
                for t in overlapping_tracks:
                    split_track_at_last_detection(t)
                active_tracks.append(create_new_track(detection))

        debug.close()

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
                    detection,
                    candidate_track,
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
        detection: Detection,
        track: Track,
        cmc: CMC,
        kf: KFState,
        ema: Embedding,
    ) -> _ComparisonResult:
        """
        Score motion (Mahalanobis d² on [cx,cy]) + appearance (χ² on L1-hist EMA).
        """
        # Early guard on excessive frame gap (keeps candidate set small)
        gap = detection.frame_idx - track.end_frame
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

    def _debug_visualize(
        self,
        detections_by_frame: Dict[FrameIndex, List[Detection]],
        frame_idx: FrameIndex,
        active_tracks: List[Track],
        kf: Dict[TrackId, KFState],
        ema: Dict[TrackId, Embedding],
        cmc: CMC,
        debug: DebugSession,
        video_properties: VideoInfo,
    ):
        if self.debug_vis:
            frame_detections_dbg: List[Detection] = detections_by_frame[frame_idx]
            sorted_active_tracks = sorted(active_tracks, key=lambda t: t.track_id)
            num_rows = len(sorted_active_tracks)
            num_cols = len(frame_detections_dbg)
            if num_rows > 0 and num_cols > 0:
                motion_matrix = np.full((num_rows, num_cols), np.nan, dtype=np.float32)
                appearance_matrix = np.full((num_rows, num_cols), np.nan, dtype=np.float32)
                for row_index, track in enumerate(sorted_active_tracks):
                    kf_state = kf.get(track.track_id, KFState.init(track.sorted_detections[0]))
                    pred = kf_state.predict_to(frame_idx, cmc)
                    track_ema = ema.get(track.track_id, track.start.embedding)
                    for col_index, detection in enumerate(frame_detections_dbg):
                        motion_matrix[row_index, col_index] = probability_from_dist(
                            pred.gating_distance(detection.bbox.center_wh, only_position=True),
                            df=2,
                        )
                        appearance_matrix[row_index, col_index] = track_ema.probability(detection.embedding)

                # Compose overlays on current frame
                frame_img = debug.get_frame(int(frame_idx))
                if frame_img is None:
                    frame_img = np.full(
                        (int(video_properties.height), int(video_properties.width), 3), 20, dtype=np.uint8
                    )
                overlays = [
                    DetectionsOverlay(detections=frame_detections_dbg),
                    KalmanOverlay(
                        kalman_states_by_track_id={tid: kf[tid] for tid in kf.keys()},
                        camera_motion_compensator=cmc,
                        target_frame_index=int(frame_idx),
                    ),
                ]
                debug.show_frame(int(frame_idx), overlays=overlays)

                # Heatmaps
                row_labels = [t.track_id for t in sorted_active_tracks]
                col_labels = list(range(num_cols))
                hm_chi2 = draw_heatmap(
                    appearance_matrix,
                    row_labels=row_labels,
                    col_labels=col_labels,
                    title='Appearance probability',
                    vmin=0.0,
                    vmax=1.0,
                )
                hm_d2 = draw_heatmap(
                    motion_matrix,
                    row_labels=row_labels,
                    col_labels=col_labels,
                    title='Motion probability',
                    vmin=0.0,
                    vmax=1.0,
                )
                debug.show(hm_chi2, hm_d2, window_name='greedy_heatmaps (higher = better)')

                # Step-by-step navigation: wait for user key, allow back with Left/comma
                debug.wait_step()
