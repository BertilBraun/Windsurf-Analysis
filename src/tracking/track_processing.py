"""
Track processing functions for merging, filtering, and smoothing detection tracks.

This module provides pure functions for processing object tracking data without
maintaining any state, making it easier to test and reason about.
"""

import logging

from settings import (
    HISTOGRAM_SIMILARITY_THRESHOLD,
    MAX_SPATIAL_DISTANCE_BB,
    MAX_TEMPORAL_DISTANCE_SECONDS,
    MIN_FRAME_PERCENTAGE,
    SMOOTHING_WINDOW_SIZE,
)
from video_io import VideoInfo
from common_types import Detection, Track, BoundingBox, TrackId, cosine_similarity


def _calculate_spatial_distance(main_track: list[Detection], other_track: list[Detection]) -> float:
    # Calculate spatial distance between bbox centers
    end = main_track[-1]
    start = other_track[0]

    end_center = end.bbox.center
    start_center = start.bbox.center

    spatial_distance = end_center.distance_to(start_center)

    # Normalize spatial distance by bbox size for scale invariance
    avg_bbox_size = (end.bbox.width + end.bbox.height + start.bbox.width + start.bbox.height) / 4
    normalized_spatial_distance = spatial_distance / max(avg_bbox_size, 1)

    return normalized_spatial_distance


def _calculate_temporal_distance(main_track: list[Detection], other_track: list[Detection]) -> float:
    return other_track[0].frame_idx - main_track[-1].frame_idx


def _interpolate_missing_boxes(track_data: list[Detection]) -> list[Detection]:
    """Interpolate bounding boxes for missing frames in a track"""
    if len(track_data) < 2:
        return track_data

    interpolated: list[Detection] = []

    for i in range(len(track_data) - 1):
        current = track_data[i]
        interpolated.append(current)

        # Check if there's a gap to the next detection
        next_detection = track_data[i + 1]
        frame_gap = next_detection.frame_idx - current.frame_idx

        # Interpolate across any gap size
        for gap_frame in range(1, frame_gap):
            # Linear interpolation factor
            alpha = gap_frame / frame_gap

            interpolated.append(current.interpolate(next_detection, alpha))

    interpolated.append(track_data[-1])
    return list(sorted(interpolated, key=lambda x: x.frame_idx))


def _smooth_track(track_data: list[Detection], window_size: int = SMOOTHING_WINDOW_SIZE) -> list[Detection]:
    """Smooth the center positions of a single track using a rolling window"""
    if len(track_data) <= 1:
        return track_data

    # Sort by frame index
    track_data.sort(key=lambda x: x.frame_idx)

    smoothed_track: list[Detection] = []

    for i, detection in enumerate(track_data):
        # Calculate original bbox dimensions
        bbox = detection.bbox
        width = bbox.width
        height = bbox.height

        # Determine the smoothing window (up to window_size frames before current)
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1

        # Get centers from the window
        centers_x = []
        centers_y = []

        for j in range(start_idx, end_idx):
            window_bbox = track_data[j].bbox
            center_x = (window_bbox.x1 + window_bbox.x2) / 2
            center_y = (window_bbox.y1 + window_bbox.y2) / 2
            centers_x.append(center_x)
            centers_y.append(center_y)

        # Calculate smoothed center (simple moving average)
        smooth_center_x = sum(centers_x) / len(centers_x)
        smooth_center_y = sum(centers_y) / len(centers_y)

        # Reconstruct bbox with smoothed center but original dimensions
        # Create new detection with smoothed bbox
        smoothed_detection = detection.copy()
        smoothed_detection.bbox = BoundingBox(
            int(smooth_center_x - width / 2),  # x1
            int(smooth_center_y - height / 2),  # y1
            int(smooth_center_x + width / 2),  # x2
            int(smooth_center_y + height / 2),  # y2
        )
        smoothed_track.append(smoothed_detection)

    return smoothed_track


def _smooth_track_centers(tracks: list[Track]) -> list[Track]:
    """Smooth the center positions of all tracks using a rolling window"""
    return [Track(track.track_id, _smooth_track(track.sorted_detections)) for track in tracks]


def _get_valid_tracks(tracks: list[Track], total_frames: int) -> list[Track]:
    """Get tracks that meet minimum frame percentage requirement"""
    logger = logging.getLogger(__name__)
    valid_tracks: list[Track] = []
    min_frames = int(MIN_FRAME_PERCENTAGE / 100 * total_frames)

    logger.info(
        f'Track analysis (min_frames required: {min_frames} out of {total_frames} total frames, {MIN_FRAME_PERCENTAGE}%):'
    )
    logger.info(f'Total tracks found: {len(tracks)}')

    for track in tracks:
        if len(track.sorted_detections) == 0:
            continue

        # Calculate track duration in frames
        first_frame = track.sorted_detections[0].frame_idx
        last_frame = track.sorted_detections[-1].frame_idx
        duration_frames = last_frame - first_frame

        if duration_frames >= min_frames:
            # Interpolate missing detections
            valid_tracks.append(Track(track.track_id, _interpolate_missing_boxes(track.sorted_detections)))

    return valid_tracks


def _relabel_tracks(tracks: list[Track]) -> list[Track]:
    """Relabel tracks from 1 to n"""
    return [Track(i, track.sorted_detections) for i, track in enumerate(tracks, start=1)]


def tracks_filtering_smoothing_relabeling(track_inputs: list[Track], video_properties: VideoInfo) -> list[Track]:
    """Complete track processing pipeline: merge, filter, and smooth tracks.

    Args:
        track_inputs: TrackerInput object containing the tracks to process
        video_properties: Video properties including total frames and fps

    Returns:
        Dictionary of processed tracks ready for video generation
    """
    logger = logging.getLogger(__name__)

    # sort all tracks by frame index
    track_inputs = [
        Track(track.track_id, list(sorted(track.sorted_detections, key=lambda x: x.frame_idx)))
        for track in track_inputs
    ]

    # First, greedily merge ALL tracks based on spatial and temporal proximity
    logger.info(f'Starting with {len(track_inputs)} tracks')

    # Then filter merged tracks for minimum duration requirement
    valid_tracks = _get_valid_tracks(track_inputs, video_properties.total_frames)

    if not valid_tracks:
        logger.warning('No merged tracks meet the minimum frame percentage requirement')
        return []

    logger.info(f'Found {len(valid_tracks)} valid tracks with duration >= {MIN_FRAME_PERCENTAGE}% of total frames')

    # Apply smoothing to reduce jittery motion
    logger.info('Smoothing track centers to reduce jittery motion...')
    smoothed_tracks = _smooth_track_centers(valid_tracks)

    # relabel tracks from 1 to n
    return _relabel_tracks(smoothed_tracks)
