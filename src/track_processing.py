"""
Track processing functions for merging, filtering, and smoothing detection tracks.

This module provides pure functions for processing object tracking data without
maintaining any state, making it easier to test and reason about.
"""

from dataclasses import dataclass
import math

from settings import (
    HISTOGRAM_SIMILARITY_THRESHOLD,
    MAX_SPATIAL_DISTANCE_BB,
    MAX_TEMPORAL_DISTANCE_SECONDS,
    MIN_FRAME_PERCENTAGE,
)
from detector import BoundingBox
from video_io import VideoInfo


@dataclass
class Track:
    frame_idx: int
    bbox: BoundingBox
    confidence: float
    hue_histogram: list[float]  # Normalized hue histogram (36 bins, 10° each)

    def copy(self) -> 'Track':
        return Track(
            self.frame_idx,
            self.bbox.copy(),
            self.confidence,
            self.hue_histogram.copy(),
        )

    def histogram_similarity(self, other: 'Track') -> float:
        """Calculate similarity between two tracks based on their hue histograms.

        Uses Bhattacharyya coefficient for histogram comparison.

        Args:
            other: Another Track to compare against

        Returns:
            Similarity score between 0.0 (no similarity) and 1.0 (identical)
        """
        if len(self.hue_histogram) != len(other.hue_histogram):
            raise ValueError('Histograms must have the same number of bins')

        # Bhattacharyya coefficient: sum of sqrt(p_i * q_i) for all bins
        similarity = sum(
            math.sqrt(self.hue_histogram[i] * other.hue_histogram[i]) for i in range(len(self.hue_histogram))
        )

        return similarity

    def interpolate(self, other: 'Track', alpha: float) -> 'Track':
        """Interpolate between this track and another track.

        Args:
            other: Target track to interpolate towards
            alpha: Interpolation factor (0.0 = this track, 1.0 = other track)

        Returns:
            New Track with interpolated values
        """
        # Interpolate frame index
        interpolated_frame_idx = int((1 - alpha) * self.frame_idx + alpha * other.frame_idx)

        # Interpolate bounding box
        interpolated_bbox = self.bbox.interpolate(other.bbox, alpha)

        # Interpolate confidence
        interpolated_confidence = (1 - alpha) * self.confidence + alpha * other.confidence
        interpolated_confidence *= 0.7  # Reduce confidence for interpolated detections

        # Interpolate hue histogram (bin by bin)
        interpolated_histogram = [
            (1 - alpha) * self.hue_histogram[i] + alpha * other.hue_histogram[i] for i in range(len(self.hue_histogram))
        ]

        return Track(interpolated_frame_idx, interpolated_bbox, interpolated_confidence, interpolated_histogram)


TrackId = int | None
TrackerInput = dict[TrackId, list[Track]]


def _tracks_overlap_temporally(track1_data: list[Track], track2_data: list[Track]) -> bool:
    """Check if two tracks have any overlapping frames"""
    track1_frames = set(det.frame_idx for det in track1_data)
    track2_frames = set(det.frame_idx for det in track2_data)
    return len(track1_frames & track2_frames) > 0


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
    # TODO if tracks overlap temporally, return inf?
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
        for track_id2 in end_track_ids:
            if track_id1 == track_id2:
                continue

            track_data1, track_data2 = tracks[track_id1], tracks[track_id2]
            temporal_distance = _calculate_temporal_distance(track_data1, track_data2)
            if (
                temporal_distance < MAX_TEMPORAL_DISTANCE_SECONDS * fps and temporal_distance > 0
            ):  # TODO aufweichen >= -XXX??
                spatial_distance = _calculate_spatial_distance(track_data1, track_data2)
                if spatial_distance < MAX_SPATIAL_DISTANCE_BB:
                    candidates.append((track_id2, spatial_distance))

        if len(candidates) == 1:
            return track_id1, candidates[0][0]

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
    print(f'Starting greedy merge with {len(tracks)} tracks...')

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
        print(f'Iteration {iteration}: Merging tracks {track_id1} and {track_id2}')

        # Merge the tracks
        merged_track = _merge_two_tracks(working_tracks[track_id1], working_tracks[track_id2])

        # Remove the individual tracks and add the merged one
        del working_tracks[track_id2]
        working_tracks[track_id1] = merged_track

    print(f'Greedy merge complete after {iteration - 1} merges: {len(tracks)} → {len(working_tracks)} tracks')
    return working_tracks


def _calculate_track_overlap(track1_data: list[Track], track2_data: list[Track]) -> float:
    """Calculate the percentage of frames that two tracks overlap"""
    track1_frames = set(det.frame_idx for det in track1_data)
    track2_frames = set(det.frame_idx for det in track2_data)

    overlap_frames = track1_frames & track2_frames

    if not overlap_frames:
        return 0.0

    # Calculate overlap as percentage of the smaller track
    smaller_track_size = min(len(track1_frames), len(track2_frames))
    overlap_percentage = len(overlap_frames) / smaller_track_size

    return overlap_percentage


def _remove_overlapping_tracks(tracks: TrackerInput, overlap_threshold: float = 0.8) -> TrackerInput:
    """Remove tracks that overlap significantly with others, keeping the longer one"""
    if len(tracks) <= 1:
        return tracks

    overlap_threshold = 10000000  # TODO

    print(f'Checking for overlapping tracks (threshold: {overlap_threshold * 100:.0f}%)...')

    track_ids = list(tracks.keys())
    tracks_to_remove = set()

    for i in range(len(track_ids)):
        for j in range(i + 1, len(track_ids)):
            track_id1, track_id2 = track_ids[i], track_ids[j]

            # Skip if either track is already marked for removal
            if track_id1 in tracks_to_remove or track_id2 in tracks_to_remove:
                continue

            overlap = _calculate_track_overlap(tracks[track_id1], tracks[track_id2])

            if overlap >= overlap_threshold:
                # Keep the longer track (more detections)
                if len(tracks[track_id1]) >= len(tracks[track_id2]):
                    tracks_to_remove.add(track_id2)
                    print(f'Removing track {track_id2} (overlaps {overlap * 100:.1f}% with longer track {track_id1})')
                else:
                    tracks_to_remove.add(track_id1)
                    print(f'Removing track {track_id1} (overlaps {overlap * 100:.1f}% with longer track {track_id2})')

    # Remove overlapping tracks
    filtered_tracks = {
        track_id: track_data for track_id, track_data in tracks.items() if track_id not in tracks_to_remove
    }

    if tracks_to_remove:
        print(f'Removed {len(tracks_to_remove)} overlapping tracks: {len(tracks)} → {len(filtered_tracks)} tracks')

    return filtered_tracks


def _smooth_track(track_data: list[Track], window_size: int = 10) -> list[Track]:
    """Smooth the center positions of a single track using a rolling window"""
    if len(track_data) <= 1:
        return track_data

    # Sort by frame index
    track_data.sort(key=lambda x: x.frame_idx)

    smoothed_track = []

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
    valid_tracks = {}
    min_frames = int(MIN_FRAME_PERCENTAGE / 100 * total_frames)

    print(
        f'Track analysis (min_frames required: {min_frames} out of {total_frames} total frames, {MIN_FRAME_PERCENTAGE}%):'
    )
    print(f'Total tracks found: {len(tracks)}')

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


def process_tracks(track_inputs: TrackerInput, video_properties: VideoInfo) -> TrackerInput:
    """Complete track processing pipeline: merge, filter, and smooth tracks.

    Args:
        track_inputs: TrackerInput object containing the tracks to process
        total_frames: Total number of frames in the video

    Returns:
        Dictionary of processed tracks ready for video generation
    """

    # sort all tracks by frame index
    track_inputs = {
        track_id: list(sorted(track_data, key=lambda x: x.frame_idx)) for track_id, track_data in track_inputs.items()
    }

    # First, greedily merge ALL tracks based on spatial and temporal proximity
    print(f'Starting with {len(track_inputs)} tracks')
    merged_tracks = _greedy_merge_tracks(track_inputs, video_properties.fps)

    # Remove tracks that still overlap significantly after merging
    merged_tracks = _remove_overlapping_tracks(merged_tracks)

    # Then filter merged tracks for minimum duration requirement
    valid_tracks = _get_valid_tracks(merged_tracks, video_properties.total_frames)

    if not valid_tracks:
        print('No merged tracks meet the minimum frame percentage requirement')
        return {}

    print(f'Found {len(valid_tracks)} valid tracks with duration >= {MIN_FRAME_PERCENTAGE}% of total frames')

    # Apply smoothing to reduce jittery motion
    print('Smoothing track centers to reduce jittery motion...')
    smoothed_tracks = _smooth_track_centers(valid_tracks)

    # relabel tracks from 1 to n
    relabeled_tracks: TrackerInput = {}
    for i, (track_id, track_data) in enumerate(smoothed_tracks.items(), start=1):
        relabeled_tracks[i] = track_data

    return relabeled_tracks
