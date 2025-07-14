from collections import defaultdict

from dataclasses import dataclass

from settings import MIN_FRAME_PERCENTAGE, MAX_MERGE_FRAME_GAP
from detector import BoundingBox


@dataclass
class Track:
    frame_idx: int
    bbox: BoundingBox
    confidence: float

    def copy(self) -> 'Track':
        return Track(self.frame_idx, self.bbox.copy(), self.confidence)


class TrackerInput:
    def __init__(self):
        self.tracks = defaultdict(list)

    def add_detection(self, frame_idx: int, track_id: int, bbox: BoundingBox, confidence: float):
        self.tracks[track_id].append(Track(frame_idx, bbox.copy(), confidence))


class TrackProcessor:
    """Handles track merging, grouping, and filtering logic"""

    def _tracks_overlap_temporally(self, track1_data: list[Track], track2_data: list[Track]) -> bool:
        """Check if two tracks have any overlapping frames"""
        track1_frames = set(det.frame_idx for det in track1_data)
        track2_frames = set(det.frame_idx for det in track2_data)
        return len(track1_frames & track2_frames) > 0

    def _calculate_track_distance(self, track1_data: list[Track], track2_data: list[Track]) -> float:
        """Calculate combined spatial and temporal distance between two tracks"""
        if self._tracks_overlap_temporally(track1_data, track2_data):
            return float('inf')  # Can't merge overlapping tracks

        # Get sorted frame indices for both tracks
        track1_frames = sorted([det.frame_idx for det in track1_data])
        track2_frames = sorted([det.frame_idx for det in track2_data])

        # Calculate temporal gap
        gap1 = track2_frames[0] - track1_frames[-1]  # track2 comes after track1
        gap2 = track1_frames[0] - track2_frames[-1]  # track1 comes after track2

        # Determine which track comes first and the temporal gap
        if gap1 > 0:  # track1 → track2
            temporal_gap = gap1
            # Get spatial positions at connection points
            track1_end = next(det for det in track1_data if det.frame_idx == track1_frames[-1])
            track2_start = next(det for det in track2_data if det.frame_idx == track2_frames[0])
            end_bbox = track1_end.bbox
            start_bbox = track2_start.bbox
        elif gap2 > 0:  # track2 → track1
            temporal_gap = gap2
            # Get spatial positions at connection points
            track2_end = next(det for det in track2_data if det.frame_idx == track2_frames[-1])
            track1_start = next(det for det in track1_data if det.frame_idx == track1_frames[0])
            end_bbox = track2_end.bbox
            start_bbox = track1_start.bbox
        else:
            return float('inf')  # Overlapping tracks

        # Calculate spatial distance between bbox centers
        end_center = end_bbox.center
        start_center = start_bbox.center

        spatial_distance = end_center.distance_to(start_center)

        # Normalize spatial distance by bbox size for scale invariance
        avg_bbox_size = (end_bbox.width + end_bbox.height + start_bbox.width + start_bbox.height) / 4
        normalized_spatial_distance = spatial_distance / max(avg_bbox_size, 1)

        # Combined distance (weighted sum of temporal and spatial components)
        # Temporal gap is in frames, spatial is normalized by bbox size
        combined_distance = temporal_gap + normalized_spatial_distance * 30  # 30 frames weight for spatial

        return combined_distance

    def _find_best_merge_candidates(self, tracks: dict[int, list[Track]]) -> tuple[tuple[int, int] | None, float]:
        """Find the pair of tracks that are closest in space and time"""
        best_distance = float('inf')
        best_pair = None

        track_ids = list(tracks.keys())

        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track_id1, track_id2 = track_ids[i], track_ids[j]
                distance = self._calculate_track_distance(tracks[track_id1], tracks[track_id2])

                if distance < best_distance:
                    best_distance = distance
                    best_pair = (track_id1, track_id2)

        return best_pair, best_distance

    def _merge_two_tracks(self, track1_data: list[Track], track2_data: list[Track]) -> list[Track]:
        """Merge two tracks by combining their detections and interpolating between endpoints"""
        # Combine all detections
        merged_detections = track1_data + track2_data

        # Sort by frame index
        merged_detections.sort(key=lambda x: x.frame_idx)

        # Apply interpolation to fill gaps
        return self._interpolate_missing_boxes(merged_detections)

    def _greedy_merge_tracks(self, tracks: dict[int, list[Track]]) -> dict[int, list[Track]]:
        """Greedy merging: repeatedly find and merge the closest track pair"""
        print(f'Starting greedy merge with {len(tracks)} tracks...')

        # Convert to working copy
        working_tracks = {track_id: track_data.copy() for track_id, track_data in tracks.items()}

        iteration = 0
        while True:
            iteration += 1

            # Find the best pair to merge
            best_pair, best_distance = self._find_best_merge_candidates(working_tracks)

            if best_pair is None or best_distance > MAX_MERGE_FRAME_GAP:
                break

            track_id1, track_id2 = best_pair
            print(f'Iteration {iteration}: Merging tracks {track_id1} and {track_id2} (distance: {best_distance:.1f})')

            # Merge the tracks
            merged_track = self._merge_two_tracks(working_tracks[track_id1], working_tracks[track_id2])

            # Remove the individual tracks and add the merged one
            del working_tracks[track_id2]
            working_tracks[track_id1] = merged_track

        print(f'Greedy merge complete after {iteration - 1} merges: {len(tracks)} → {len(working_tracks)} tracks')
        return working_tracks

    def _calculate_track_overlap(self, track1_data: list[Track], track2_data: list[Track]) -> float:
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

    def _remove_overlapping_tracks(
        self, tracks: dict[int, list[Track]], overlap_threshold: float = 0.8
    ) -> dict[int, list[Track]]:
        """Remove tracks that overlap significantly with others, keeping the longer one"""
        if len(tracks) <= 1:
            return tracks

        print(f'Checking for overlapping tracks (threshold: {overlap_threshold * 100:.0f}%)...')

        track_ids = list(tracks.keys())
        tracks_to_remove = set()

        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                track_id1, track_id2 = track_ids[i], track_ids[j]

                # Skip if either track is already marked for removal
                if track_id1 in tracks_to_remove or track_id2 in tracks_to_remove:
                    continue

                overlap = self._calculate_track_overlap(tracks[track_id1], tracks[track_id2])

                if overlap >= overlap_threshold:
                    # Keep the longer track (more detections)
                    if len(tracks[track_id1]) >= len(tracks[track_id2]):
                        tracks_to_remove.add(track_id2)
                        print(
                            f'Removing track {track_id2} (overlaps {overlap * 100:.1f}% with longer track {track_id1})'
                        )
                    else:
                        tracks_to_remove.add(track_id1)
                        print(
                            f'Removing track {track_id1} (overlaps {overlap * 100:.1f}% with longer track {track_id2})'
                        )

        # Remove overlapping tracks
        filtered_tracks = {
            track_id: track_data for track_id, track_data in tracks.items() if track_id not in tracks_to_remove
        }

        if tracks_to_remove:
            print(f'Removed {len(tracks_to_remove)} overlapping tracks: {len(tracks)} → {len(filtered_tracks)} tracks')

        return filtered_tracks

    def _smooth_track_centers(self, tracks: dict[int, list[Track]]) -> dict[int, list[Track]]:
        """Smooth the center positions of bounding boxes over a rolling window"""
        return {track_id: self._smooth_track(track_data) for track_id, track_data in tracks.items()}

    def _smooth_track(self, track_data: list[Track], window_size: int = 10) -> list[Track]:
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

    def _interpolate_missing_boxes(self, track_data: list[Track]) -> list[Track]:
        """Interpolate bounding boxes for missing frames in a track"""
        if len(track_data) < 2:
            return track_data

        # Sort by frame index
        track_data.sort(key=lambda x: x.frame_idx)

        interpolated: list[Track] = []

        for i in range(len(track_data) - 1):
            current = track_data[i]
            interpolated.append(current)

            # Check if there's a gap to the next detection
            next_detection = track_data[i + 1]
            frame_gap = next_detection.frame_idx - current.frame_idx

            # Interpolate across any gap size
            for gap_frame in range(1, frame_gap):
                interp_frame_idx = current.frame_idx + gap_frame

                # Linear interpolation factor
                alpha = gap_frame / frame_gap

                # Interpolate bounding box
                interp_bbox = current.bbox.interpolate(next_detection.bbox, alpha)

                # Interpolate confidence (lower for interpolated)
                interp_confidence = (1 - alpha) * current.confidence + alpha * next_detection.confidence
                interp_confidence *= 0.7  # Reduce confidence for interpolated detections

                interpolated.append(Track(interp_frame_idx, interp_bbox, interp_confidence))

        interpolated.append(track_data[-1])
        return sorted(interpolated, key=lambda x: x.frame_idx)

    def _get_valid_tracks(self, tracks: dict[int, list[Track]], total_frames: int) -> dict[int, list[Track]]:
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
                valid_tracks[track_id] = self._interpolate_missing_boxes(detections)

        return valid_tracks

    def process_tracks(self, track_inputs: TrackerInput, total_frames: int) -> dict[int, list[Track]]:
        """Complete track processing pipeline: merge, filter, and smooth tracks.

        Args:
            track_inputs: TrackerInput object containing the tracks to process
            total_frames: Total number of frames in the video

        Returns:
            Dictionary of processed tracks ready for video generation
        """
        # First, greedily merge ALL tracks based on spatial and temporal proximity
        print(f'Starting with {len(track_inputs.tracks)} tracks')
        merged_tracks = self._greedy_merge_tracks(track_inputs.tracks)

        # Remove tracks that still overlap significantly after merging
        merged_tracks = self._remove_overlapping_tracks(merged_tracks)

        # Then filter merged tracks for minimum duration requirement
        valid_tracks = self._get_valid_tracks(merged_tracks, total_frames)

        if not valid_tracks:
            print('No merged tracks meet the minimum frame percentage requirement')
            return {}

        print(f'Found {len(valid_tracks)} valid tracks with duration >= {MIN_FRAME_PERCENTAGE}% of total frames')

        # Apply smoothing to reduce jittery motion
        print('Smoothing track centers to reduce jittery motion...')
        smoothed_tracks = self._smooth_track_centers(valid_tracks)

        return smoothed_tracks
