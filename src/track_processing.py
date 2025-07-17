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
from common_types import Track, BoundingBox, TrackId, TrackerInput


def _calculate_spatial_distance(main_track: list[Track], other_track: list[Track]) -> float:
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


def _calculate_temporal_distance(main_track: list[Track], other_track: list[Track]) -> float:
    return other_track[0].frame_idx - main_track[-1].frame_idx


def _find_best_merge_candidates(tracks: TrackerInput, fps: int) -> tuple[TrackId, TrackId] | None:
    """Find the pair of tracks that are closest in space and time"""

    all_candidates: dict[TrackId, list[tuple[TrackId, float]]] = {}

    start_track_ids = list(tracks.keys())
    start_track_ids.sort(key=lambda x: tracks[x][0].frame_idx)

    end_track_ids = list(tracks.keys())
    end_track_ids.sort(key=lambda x: tracks[x][-1].frame_idx)

    for track_id1 in start_track_ids:
        candidates: list[tuple[TrackId, float]] = []
        greedy_candidates: list[tuple[TrackId, float]] = []

        for track_id2 in end_track_ids:
            if track_id1 == track_id2:
                continue

            track_data1, track_data2 = tracks[track_id1], tracks[track_id2]
            spatial_distance = _calculate_spatial_distance(track_data1, track_data2)

            is_close = spatial_distance < MAX_SPATIAL_DISTANCE_BB

            if is_close:
                temporal_distance = _calculate_temporal_distance(track_data1, track_data2)
                max_temporal_distance = MAX_TEMPORAL_DISTANCE_SECONDS * fps

                is_within_temporal_distance = temporal_distance < max_temporal_distance and temporal_distance > 0
                is_within_greedy_temporal_distance = temporal_distance < 0.25 * fps and temporal_distance > 0

                if is_within_temporal_distance:
                    candidates.append((track_id2, spatial_distance))
                if is_within_greedy_temporal_distance:
                    greedy_candidates.append((track_id2, spatial_distance))

        if len(greedy_candidates) == 1:
            logger = logging.getLogger(__name__)
            logger.debug(f'Found greedy candidate: {track_id1} and {greedy_candidates[0][0]}')
            return track_id1, greedy_candidates[0][0]

        all_candidates[track_id1] = candidates

    for track_id1, candidates in all_candidates.items():
        end_track = tracks[track_id1][-1]

        def get_histogram_similarity(candidate: tuple[TrackId, float]) -> float:
            start_track = tracks[candidate[0]][0]
            return end_track.histogram_similarity(start_track)

        candidates.sort(key=get_histogram_similarity)

        if candidates and get_histogram_similarity(candidates[0]) > HISTOGRAM_SIMILARITY_THRESHOLD:
            return track_id1, candidates[0][0]

    return None


def _interpolate_missing_boxes(track_data: list[Track]) -> list[Track]:
    """Interpolate bounding boxes for missing frames in a track"""
    if len(track_data) < 2:
        return track_data

    interpolated: list[Track] = []

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


def _merge_two_tracks(track1_data: list[Track], track2_data: list[Track]) -> list[Track]:
    """Merge two tracks by combining their detections and interpolating between endpoints"""
    # Combine all detections
    merged_detections = track1_data + track2_data

    # Sort by frame index
    merged_detections.sort(key=lambda x: x.frame_idx)

    # Apply interpolation to fill gaps
    return _interpolate_missing_boxes(merged_detections)


def _greedy_merge_tracks(tracks: TrackerInput, fps: int) -> TrackerInput:
    """Greedy merging: repeatedly find and merge the closest track pair"""
    logger = logging.getLogger(__name__)
    logger.info(f'Starting greedy merge with {len(tracks)} tracks...')

    # Convert to working copy
    working_tracks = {track_id: track_data.copy() for track_id, track_data in tracks.items()}

    iteration = 0
    while True:
        iteration += 1

        # Find the best pair to merge
        best_pair = _find_best_merge_candidates(working_tracks, fps)

        if best_pair is None:
            break

        track_id1, track_id2 = best_pair
        logger.info(f'Iteration {iteration}: Merging tracks {track_id1} and {track_id2}')

        # Merge the tracks
        merged_track = _merge_two_tracks(working_tracks[track_id1], working_tracks[track_id2])

        # Remove the individual tracks and add the merged one
        del working_tracks[track_id2]
        working_tracks[track_id1] = merged_track

    logger.info(f'Greedy merge complete after {iteration - 1} merges: {len(tracks)} → {len(working_tracks)} tracks')
    return working_tracks


def _smooth_track(track_data: list[Track], window_size: int = SMOOTHING_WINDOW_SIZE) -> list[Track]:
    """Smooth the center positions of a single track using a rolling window"""
    if len(track_data) <= 1:
        return track_data

    # Sort by frame index
    track_data.sort(key=lambda x: x.frame_idx)

    smoothed_track: list[Track] = []

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


def _smooth_track_centers(tracks: TrackerInput) -> TrackerInput:
    """Smooth the center positions of all tracks using a rolling window"""
    return {track_id: _smooth_track(track_data) for track_id, track_data in tracks.items()}


def _get_valid_tracks(tracks: TrackerInput, total_frames: int) -> TrackerInput:
    """Get tracks that meet minimum frame percentage requirement"""
    logger = logging.getLogger(__name__)
    valid_tracks = {}
    min_frames = int(MIN_FRAME_PERCENTAGE / 100 * total_frames)

    logger.info(
        f'Track analysis (min_frames required: {min_frames} out of {total_frames} total frames, {MIN_FRAME_PERCENTAGE}%):'
    )
    logger.info(f'Total tracks found: {len(tracks)}')

    for track_id, detections in tracks.items():
        if len(detections) == 0:
            continue

        # Sort detections by frame
        detections.sort(key=lambda x: x.frame_idx)

        # Calculate track duration in frames
        first_frame = detections[0].frame_idx
        last_frame = detections[-1].frame_idx
        duration_frames = last_frame - first_frame

        if duration_frames >= min_frames:
            # Interpolate missing detections
            valid_tracks[track_id] = _interpolate_missing_boxes(detections)

    return valid_tracks


def _relabel_tracks(tracks: TrackerInput) -> TrackerInput:
    """Relabel tracks from 1 to n"""
    return {i: track_data for i, (track_id, track_data) in enumerate(tracks.items(), start=1)}


def process_tracks(track_inputs: TrackerInput, video_properties: VideoInfo) -> TrackerInput:
    """Complete track processing pipeline: merge, filter, and smooth tracks.

    Args:
        track_inputs: TrackerInput object containing the tracks to process
        video_properties: Video properties including total frames and fps

    Returns:
        Dictionary of processed tracks ready for video generation
    """
    logger = logging.getLogger(__name__)

    # sort all tracks by frame index
    track_inputs = {
        track_id: list(sorted(track_data, key=lambda x: x.frame_idx)) for track_id, track_data in track_inputs.items()
    }

    # First, greedily merge ALL tracks based on spatial and temporal proximity
    logger.info(f'Starting with {len(track_inputs)} tracks')
    merged_tracks = _greedy_merge_tracks(track_inputs, video_properties.fps)

    # Then filter merged tracks for minimum duration requirement
    valid_tracks = _get_valid_tracks(merged_tracks, video_properties.total_frames)

    if not valid_tracks:
        logger.warning('No merged tracks meet the minimum frame percentage requirement')
        return {}

    logger.info(f'Found {len(valid_tracks)} valid tracks with duration >= {MIN_FRAME_PERCENTAGE}% of total frames')

    # Apply smoothing to reduce jittery motion
    logger.info('Smoothing track centers to reduce jittery motion...')
    smoothed_tracks = _smooth_track_centers(valid_tracks)

    # relabel tracks from 1 to n
    return _relabel_tracks(smoothed_tracks)
