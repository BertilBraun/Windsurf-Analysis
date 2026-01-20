"""
Track processing functions for merging, filtering, and smoothing detection tracks.

This module provides pure functions for processing object tracking data without
maintaining any state, making it easier to test and reason about.
"""

import logging
from bisect import bisect_left
from typing import Sequence, Optional
import numpy as np

from ..tracking.tracking import Tracker
from ..visualization.stabilize import Transform

from ..settings import (
    MIN_FRAME_PERCENTAGE,
    TRACK_RTS_ENABLE_BACKWARD_SMOOTHER,
    TRACK_RTS_MEAS_STD_WEIGHT_POS,
    TRACK_RTS_MEAS_STD_WEIGHT_SIZE,
    TRACK_RTS_PROC_STD_WEIGHT_POS,
    TRACK_RTS_PROC_STD_WEIGHT_VEL,
)
from ..util.video_io import VideoInfo
from ..common_types import (
    Detection,
    Track,
    BoundingBox,
    FrameIndex,
    Point,
    Keypoint,
    RenderableDetection,
    RenderableTrack,
    interpolate,
)
from ..motion.kalman_filter import KFState, _KalmanFilter, KF
from ..motion.cmc import CMC


class TrackPostProcessing:
    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        trackers: Sequence[Tracker] = [TrackFiltering(), TrackRTSSmoothing(), TrackRelabeling()]
        for tracker in trackers:
            tracks = tracker.track(tracks, video_properties, transforms)
        return tracks


class TrackFiltering:
    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        """Filter tracks for minimum duration requirement"""
        return _get_valid_tracks(tracks, video_properties.total_frames)


class TrackRelabeling:
    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        """Relabel tracks from 1 to n"""
        return [Track(i, track.sorted_detections) for i, track in enumerate(tracks, start=1)]


class TrackRTSSmoothing:
    """
    Combined interpolation and smoothing using Rauch-Tung-Striebel (RTS) smoother.

    This replaces both TrackInterpolation and TrackSmoothing with a single,
    theoretically optimal approach that:
    1. Runs a forward Kalman filter pass with camera motion compensation
    2. Runs a backward RTS smoothing pass
    3. Fills gaps and smooths trajectories simultaneously
    """

    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        """Apply RTS smoothing to all tracks with camera motion compensation"""
        cmc = CMC(transforms)
        return [Track(track.track_id, _rts_smooth_track(track.sorted_detections, cmc)) for track in tracks]


def _get_valid_tracks(tracks: list[Track], total_frames: int) -> list[Track]:
    """Get tracks that meet minimum frame percentage requirement"""
    logger = logging.getLogger(__name__)
    valid_tracks: list[Track] = []
    min_frames = int(MIN_FRAME_PERCENTAGE / 100 * total_frames)

    logger.info(
        f'Track analysis (min_frames required: {min_frames} out of {total_frames} total frames, {MIN_FRAME_PERCENTAGE}%):'
    )

    for track in tracks:
        if len(track.sorted_detections) == 0:
            continue

        # Calculate track duration in frames
        duration_frames = track.end_frame - track.start_frame

        if duration_frames < min_frames:
            # if the track is too short, it is likely a false positive
            continue

        if len(track.sorted_detections) < duration_frames * 0.3:
            # if less than 30% of the track is actually based on detections, it is likely a false positive
            continue

        valid_tracks.append(track)

    return valid_tracks


# ========================================================================================
# RTS Smoothing Implementation
# ========================================================================================


def _rts_smooth_track(detections: list[Detection], cmc: CMC) -> list[Detection]:
    """
    Apply RTS (Rauch-Tung-Striebel) smoothing to a track.

    This function:
    1. Runs a forward Kalman filter pass through all detections
    2. Applies camera motion compensation at each step
    3. Stores filtered and predicted states
    4. Runs a backward RTS smoothing pass
    5. Generates dense, smoothed detections for all frames in the track span

    Args:
        detections: List of detections (may have gaps)
        cmc: Camera motion compensator

    Returns:
        Dense list of smoothed detections covering all frames from start to end
    """
    if len(detections) < 2:
        return detections

    # Sort detections by frame
    detections = sorted(detections, key=lambda d: d.frame_idx)
    start_frame = detections[0].frame_idx
    end_frame = detections[-1].frame_idx
    N = end_frame - start_frame + 1

    # Initialize Kalman filter state from first detection
    state = KFState.init(
        detections[0],
        _KalmanFilter(
            proc_std_weight_pos=TRACK_RTS_PROC_STD_WEIGHT_POS,
            proc_std_weight_vel=TRACK_RTS_PROC_STD_WEIGHT_VEL,
            meas_std_weight_pos=TRACK_RTS_MEAS_STD_WEIGHT_POS,
            meas_std_weight_size=TRACK_RTS_MEAS_STD_WEIGHT_SIZE,
        ),
    )

    # Storage for forward pass results
    # We store state for every frame in the track span, not just detection frames
    mu_filt = np.zeros((N, 8))  # filtered means
    P_filt = np.zeros((N, 8, 8))  # filtered covariances
    mu_pred = np.zeros((N, 8))  # predicted means (before update)
    P_pred = np.zeros((N, 8, 8))  # predicted covariances (before update)

    # Storage for transition matrices
    Fs = np.zeros((N - 1, 8, 8))

    # Map frame indices to detections for quick lookup
    detection_dict = {d.frame_idx: d for d in detections}
    assert len(detection_dict) == len(detections), 'Mapping had duplicate frame index'

    # Forward pass: Kalman filter with camera motion compensation
    for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
        if i == 0:
            # First frame: store initial state
            mu_pred[i] = state.mean
            P_pred[i] = state.cov
            mu_filt[i] = state.mean
            P_filt[i] = state.cov
        else:
            # Calculate transition matrix for (i-1) -> i
            # This matches the operation in state.predict_to -> advance_state_to_frame
            # x_pred = A_cmc @ F_kf @ x_prev
            prev_frame_idx = frame_idx - 1
            F_kf = KF._F_gap(1)

            # CMC transition
            if prev_frame_idx in cmc._transforms_dict:
                transform = cmc._transforms_dict[prev_frame_idx]
                A_cmc, _ = cmc._build_A_T(transform)
                F_total = A_cmc @ F_kf
            else:
                # Fallback if no transform available
                F_total = F_kf

            Fs[i - 1] = F_total

            # Predict to current frame (includes CMC)
            predicted_state = state.predict_to(frame_idx, cmc)
            mu_pred[i] = predicted_state.mean
            P_pred[i] = predicted_state.cov

            # Update if we have a detection at this frame
            if frame_idx in detection_dict:
                state = predicted_state.update_to_det(detection_dict[frame_idx], cmc)
                mu_filt[i] = state.mean
                P_filt[i] = state.cov
            else:
                # No detection: filtered state = predicted state
                state = predicted_state
                mu_filt[i] = predicted_state.mean
                P_filt[i] = predicted_state.cov

    if TRACK_RTS_ENABLE_BACKWARD_SMOOTHER:
        # Backward pass: RTS smoothing
        mu_smooth, P_smooth = _rts_smoother(mu_filt, P_filt, mu_pred, P_pred, Fs)
    else:
        # Forward-only filtering (less "global" smoothing; avoids future-looking corrections).
        mu_smooth, P_smooth = mu_filt, P_filt

    # Generate smoothed detections for all frames
    smoothed_detections: list[Detection] = []
    for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
        # Extract position and size from smoothed state
        cx, cy, w, h = mu_smooth[i, :4]
        w, h = max(w, 1), max(h, 1)

        # Create bounding box from smoothed state
        smoothed_bbox = BoundingBox.from_center_wh(cx, cy, w, h)

        # Use original detection if available, otherwise create interpolated one
        if frame_idx in detection_dict:
            original_det = detection_dict[frame_idx]
            neighborhood: list[tuple[Detection | None, float]] = [
                (detection_dict[frame_idx - 2] if frame_idx - 2 in detection_dict else None, 0),
                (detection_dict[frame_idx - 1] if frame_idx - 1 in detection_dict else None, 1),
                (original_det, 4),
                (detection_dict[frame_idx + 1] if frame_idx + 1 in detection_dict else None, 0.5),
                (detection_dict[frame_idx + 2] if frame_idx + 2 in detection_dict else None, 0),
            ]

            # Baseline: weighted average in image coordinates (as floats).
            img_centers: list[tuple[Point, float]] = []
            for det, weight in neighborhood:
                if det is None or weight <= 0:
                    continue
                img_centers.append((det.bbox.center, float(weight)))

            neighborhood_center = sum((xy * w for xy, w in img_centers), start=Point(0, 0)) / sum(
                w for _xy, w in img_centers
            )

            neighborhood_bbox_width_list = [
                detection_dict[frame_idx + i].bbox.width for i in range(-15, 16) if frame_idx + i in detection_dict
            ]
            neighborhood_bbox_width = sum(neighborhood_bbox_width_list) / len(neighborhood_bbox_width_list)
            neighborhood_bbox_height_list = [
                detection_dict[frame_idx + i].bbox.height for i in range(-15, 16) if frame_idx + i in detection_dict
            ]
            neighborhood_bbox_height = sum(neighborhood_bbox_height_list) / len(neighborhood_bbox_height_list)
            neighborhood_bbox = BoundingBox.from_center_wh(
                neighborhood_center.x,
                neighborhood_center.y,
                neighborhood_bbox_width,
                neighborhood_bbox_height,
            )
            neighborhood_bbox_1 = sum(
                (d.bbox * weight for d, weight in neighborhood if d is not None), start=BoundingBox(0, 0, 0, 0)
            ) / sum(weight for d, weight in neighborhood if d is not None)

            smoothed_det = Detection(
                bbox=neighborhood_bbox,
                embedding=original_det.embedding,
                confidence=original_det.confidence,
                frame_idx=frame_idx,
                boom=original_det.boom,
                mast_tip=original_det.mast_tip,
                interpolated=False,
            )
        else:
            # Interpolate embedding and confidence from nearest detections
            embedding, confidence, boom, mast_tip = _interpolate_detection_attributes(frame_idx, detections)
            smoothed_det = Detection(
                bbox=smoothed_bbox,
                embedding=embedding,
                confidence=confidence,
                frame_idx=frame_idx,
                boom=boom,
                mast_tip=mast_tip,
                interpolated=True,
            )

        smoothed_detections.append(smoothed_det)

    return smoothed_detections


def _rts_smoother(
    mu_filt: np.ndarray,
    P_filt: np.ndarray,
    mu_pred: np.ndarray,
    P_pred: np.ndarray,
    Fs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RTS smoother for linear Gaussian systems.

    Inputs:
      mu_filt:  [N, n]  filtered means AFTER update at each k
      P_filt:   [N, n, n] filtered covariances AFTER update
      mu_pred:  [N, n]  predicted means BEFORE update (k entry is μ_{k|k-1})
      P_pred:   [N, n, n] predicted covariances BEFORE update
      Fs:       [N-1, n, n] transition matrices F_k (x_{k+1} = F_k x_k + ...)

    Returns:
      mu_smooth, P_smooth arrays of same shape as inputs
    """
    N, n = mu_filt.shape
    mu_smooth = mu_filt.copy()
    P_smooth = P_filt.copy()

    # Backward pass: k = N-2, N-3, ..., 0
    for k in range(N - 2, -1, -1):
        # Smoothing gain G_k = P_k @ F_k^T @ (P_{k+1|k})^{-1}
        # Use pseudo-inverse for numerical stability
        G = P_filt[k] @ Fs[k].T @ np.linalg.pinv(P_pred[k + 1])

        # Mean correction: μ_k^s = μ_k + G @ (μ_{k+1}^s − μ_{k+1|k})
        mu_smooth[k] += G @ (mu_smooth[k + 1] - mu_pred[k + 1])

        # Covariance correction: P_k^s = P_k + G @ (P_{k+1}^s − P_{k+1|k}) @ G^T
        P_smooth[k] += G @ (P_smooth[k + 1] - P_pred[k + 1]) @ G.T

        # Ensure symmetry
        P_smooth[k] = 0.5 * (P_smooth[k] + P_smooth[k].T)

    return mu_smooth, P_smooth


def _interpolate_detection_attributes(frame_idx: FrameIndex, detections: list[Detection]):
    """
    Interpolate embedding and confidence for a frame without a detection.

    Uses binary search to find the nearest detections before and after,
    then performs linear interpolation.

    Args:
        frame_idx: Frame index to interpolate at
        detections: Sorted list of detections

    Returns:
        tuple: (embedding, confidence) where embedding is an Embedding object
    """
    # Binary search to find insertion point
    # detections are already sorted by frame_idx
    idx = bisect_left([d.frame_idx for d in detections], frame_idx)

    # Handle edge cases
    if idx == 0:
        # Before all detections: use first
        first = detections[0]
        return (
            first.embedding,
            first.confidence,
            Keypoint(point=first.boom.point, conf=0.0),
            Keypoint(point=first.mast_tip.point, conf=0.0),
        )

    if idx >= len(detections):
        # After all detections: use last
        last = detections[-1]
        return (
            last.embedding,
            last.confidence,
            Keypoint(point=last.boom.point, conf=0.0),
            Keypoint(point=last.mast_tip.point, conf=0.0),
        )

    # Normal case: interpolate between detections[idx-1] and detections[idx]
    before_det = detections[idx - 1]
    after_det = detections[idx]

    # Linear interpolation factor
    total_gap = after_det.frame_idx - before_det.frame_idx
    alpha = (frame_idx - before_det.frame_idx) / total_gap

    # Use the Embedding protocol's interpolate method
    embedding = before_det.embedding.interpolate(after_det.embedding, alpha)

    # Interpolate confidence
    confidence = before_det.confidence * (1 - alpha) + after_det.confidence * alpha

    boom_pt = before_det.boom.point.interpolate(after_det.boom.point, alpha)
    mast_tip_pt = before_det.mast_tip.point.interpolate(after_det.mast_tip.point, alpha)

    return embedding, confidence, Keypoint(point=boom_pt, conf=0.0), Keypoint(point=mast_tip_pt, conf=0.0)


def prepare_renderable_tracks(
    tracks: list[Track],
    *,
    kp_conf_thresh: float = 0.3,
    missing_grace_frames: int = 5,
    mast_smooth_radius: int = 15,
) -> list[RenderableTrack]:
    """
    Post-processing step for the rendering pipeline.

    Produces RenderableTrack/RenderableDetection objects with stable per-frame:
      - anchor (Point)
      - scale (float)  (normalized so mast length is constant per track)

    Rules:
      - Anchor prefers boom keypoint when confidence >= kp_conf_thresh.
      - If boom is missing for >missing_grace_frames, fallback to bbox.center.
      - Short gaps (<=missing_grace_frames) between two visible boom keypoints are linearly interpolated.
      - Mast length prefers dist(boom, mast_tip) when both are confident; after long gaps fallback uses bbox top edge.
      - Anchor smoothing: fixed weighted window [prev:1, curr:4, next:0.5].
      - Mast smoothing: mean over a +/- mast_smooth_radius window.
    """

    def _dist(a: Point, b: Point) -> float:
        return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)

    def _anchor_weighted(points: list[Point]) -> list[Point]:
        out: list[Point] = []
        for i in range(len(points)):
            acc_x = 0.0
            acc_y = 0.0
            acc_w = 0.0
            if i - 1 >= 0:
                acc_x += points[i - 1].x * 1.0
                acc_y += points[i - 1].y * 1.0
                acc_w += 1.0
            acc_x += points[i].x * 4.0
            acc_y += points[i].y * 4.0
            acc_w += 4.0
            if i + 1 < len(points):
                acc_x += points[i + 1].x * 0.5
                acc_y += points[i + 1].y * 0.5
                acc_w += 0.5
            if acc_w <= 0.0:
                out.append(points[i])
            else:
                out.append(Point(int(round(acc_x / acc_w)), int(round(acc_y / acc_w))))
        return out

    def _smooth_mean(values: list[float], radius: int) -> list[float]:
        r = max(0, int(radius))
        out: list[float] = []
        for i in range(len(values)):
            lo = max(0, i - r)
            hi = min(len(values) - 1, i + r)
            window = values[lo : hi + 1]
            out.append(float(sum(window) / max(1, len(window))))
        return out

    def _median(values: list[float]) -> float:
        if not values:
            return 1.0
        s = sorted(values)
        mid = len(s) // 2
        if len(s) % 2 == 1:
            return float(s[mid])
        return float((s[mid - 1] + s[mid]) / 2.0)

    out_tracks: list[RenderableTrack] = []
    for track in tracks:
        dets = track.sorted_detections
        if not dets:
            out_tracks.append(RenderableTrack(track_id=track.track_id, sorted_detections=[]))
            continue

        bbox_centers = [d.bbox.center for d in dets]
        boom_vis = [float(d.boom.conf) >= float(kp_conf_thresh) for d in dets]
        mast_vis = [float(d.mast_tip.conf) >= float(kp_conf_thresh) for d in dets]

        # -------------------- Anchor fill --------------------
        visible_idxs = [i for i, v in enumerate(boom_vis) if v]
        if not visible_idxs:
            anchors = bbox_centers
        else:
            anchors: list[Point] = [Point(0, 0) for _ in dets]

            # default to bbox center
            for i in range(len(dets)):
                anchors[i] = bbox_centers[i]

            # place visible
            for i in visible_idxs:
                anchors[i] = dets[i].boom.point

            # interpolate short gaps between visible anchors
            for a_i, b_i in zip(visible_idxs[:-1], visible_idxs[1:]):
                gap = b_i - a_i - 1
                if gap <= 0:
                    continue
                if gap <= int(missing_grace_frames):
                    p0 = dets[a_i].boom.point
                    p1 = dets[b_i].boom.point
                    for k in range(1, gap + 1):
                        t = float(k) / float(gap + 1)
                        anchors[a_i + k] = p0.interpolate(p1, t)

            # trailing missing: hold last visible up to grace, then bbox center
            last = visible_idxs[-1]
            for i in range(last + 1, len(dets)):
                if (i - last) <= int(missing_grace_frames):
                    anchors[i] = anchors[last]
                else:
                    anchors[i] = bbox_centers[i]

            # leading missing: keep bbox centers (no previous anchor)

        anchors = _anchor_weighted(anchors)

        # -------------------- Mast length fill --------------------
        len_vis = [boom_vis[i] and mast_vis[i] for i in range(len(dets))]
        raw_len: list[Optional[float]] = [
            _dist(dets[i].boom.point, dets[i].mast_tip.point) if len_vis[i] else None for i in range(len(dets))
        ]

        def bbox_fallback_len(i: int) -> float:
            return float(abs(int(anchors[i].y) - int(dets[i].bbox.y1)))

        visible_len_idxs = [i for i, v in enumerate(len_vis) if v]
        if not visible_len_idxs:
            filled_len = [bbox_fallback_len(i) for i in range(len(dets))]
        else:
            filled_len: list[float] = [bbox_fallback_len(i) for i in range(len(dets))]
            for i in visible_len_idxs:
                assert raw_len[i] is not None
                filled_len[i] = float(raw_len[i])

            # interpolate short gaps between visible lengths
            for a_i, b_i in zip(visible_len_idxs[:-1], visible_len_idxs[1:]):
                gap = b_i - a_i - 1
                if gap <= 0:
                    continue
                if gap <= int(missing_grace_frames):
                    v0 = float(filled_len[a_i])
                    v1 = float(filled_len[b_i])
                    for k in range(1, gap + 1):
                        t = float(k) / float(gap + 1)
                        filled_len[a_i + k] = float(interpolate(v0, v1, t))

            # trailing missing: hold last visible up to grace, then bbox fallback
            last = visible_len_idxs[-1]
            for i in range(last + 1, len(dets)):
                if (i - last) <= int(missing_grace_frames):
                    filled_len[i] = float(filled_len[last])
                else:
                    filled_len[i] = bbox_fallback_len(i)

        smoothed_len = _smooth_mean(filled_len, int(mast_smooth_radius))
        ref_candidates = [smoothed_len[i] for i in range(len(dets)) if len_vis[i]]
        ref_len = _median(ref_candidates if ref_candidates else smoothed_len)
        ref_len = max(1.0, float(ref_len))

        scales = [float(ref_len / max(1.0, float(l))) for l in smoothed_len]

        render_dets: list[RenderableDetection] = []
        for i, d in enumerate(dets):
            render_dets.append(
                RenderableDetection(
                    bbox=d.bbox,
                    confidence=float(d.confidence),
                    frame_idx=int(d.frame_idx),
                    interpolated=bool(d.interpolated),
                    boom=d.boom,
                    mast_tip=d.mast_tip,
                    anchor=anchors[i],
                    scale=float(scales[i]),
                )
            )

        out_tracks.append(RenderableTrack(track_id=track.track_id, sorted_detections=render_dets))

    return out_tracks
