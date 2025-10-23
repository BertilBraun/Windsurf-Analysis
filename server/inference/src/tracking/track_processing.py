"""
Track processing functions for merging, filtering, and smoothing detection tracks.

This module provides pure functions for processing object tracking data without
maintaining any state, making it easier to test and reason about.
"""

import logging

from server.inference.src.visualization.stabilize import Transform

from ..settings import MIN_FRAME_PERCENTAGE, SMOOTHING_WINDOW_SIZE
from ..util.video_io import VideoInfo
from ..common_types import Detection, Track, BoundingBox


class TrackFiltering:
    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        """Filter tracks for minimum duration requirement"""
        return _get_valid_tracks(tracks, video_properties.total_frames)


class TrackInterpolation:
    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        """Interpolate missing detections"""
        return [Track(track.track_id, _interpolate_missing_boxes(track.sorted_detections)) for track in tracks]


class TrackSmoothing:
    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        """Smooth the center positions of all tracks using a rolling window"""
        return [Track(track.track_id, _smooth_track(track.sorted_detections)) for track in tracks]


class TrackRelabeling:
    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        """Relabel tracks from 1 to n"""
        return [Track(i, track.sorted_detections) for i, track in enumerate(tracks, start=1)]


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

            interpolated.append(current.interpolate(next_detection, alpha, current.frame_idx + gap_frame))

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

        if duration_frames >= min_frames:
            valid_tracks.append(track)

    return valid_tracks
