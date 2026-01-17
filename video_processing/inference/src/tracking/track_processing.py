"""
Track processing functions for merging, filtering, and smoothing detection tracks.

This module provides pure functions for processing object tracking data without
maintaining any state, making it easier to test and reason about.
"""

import logging
from bisect import bisect_left
from typing import Sequence
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
from ..common_types import Detection, Track, BoundingBox, FrameIndex, Point
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
                interpolated=False,
            )
        else:
            # Interpolate embedding and confidence from nearest detections
            embedding, confidence = _interpolate_detection_attributes(frame_idx, detections)
            smoothed_det = Detection(
                bbox=smoothed_bbox,
                embedding=embedding,
                confidence=confidence,
                frame_idx=frame_idx,
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
        return detections[0].embedding, detections[0].confidence

    if idx >= len(detections):
        # After all detections: use last
        return detections[-1].embedding, detections[-1].confidence

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

    return embedding, confidence
