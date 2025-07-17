"""
Surfer tracking module for aggregating detections across frames.

This module handles the collection and organization of YOLO detections
into coherent tracks representing individual surfers.
"""

import logging
from collections import defaultdict

import numpy as np

import track_processing
from common_types import Detection, TrackDetection, BoundingBox, Point, Track
from video_io import get_video_properties
from track_processing import TrackId, TrackerInput



class SurferTracker:
    """Aggregates YOLO detections into tracks for individual surfers"""

    def __init__(self):
        self.track_inputs = defaultdict(list)
        # from frame idx to detections
        self.detections: dict[int, list[Detection]] = defaultdict(list)

    def add_detection(self, frame_idx: int, detection: Detection):
        self.detections[frame_idx].append(detection)

    def add_track_detection(self, frame_idx: int, detection: TrackDetection, frame: np.ndarray):
        """Add a detection for a specific track at a given frame"""

        # Calculate hue histogram from bounding box region (simplified for now)
        hue_histogram = self._calculate_simple_hue_histogram(frame, detection.bbox)

        # Create a track entry
        track_entry = Track(
            frame_idx=frame_idx, bbox=detection.bbox, confidence=detection.confidence, hue_histogram=hue_histogram
        )

        self.track_inputs[detection.track_id].append(track_entry)

    def _calculate_simple_hue_histogram(self, frame: np.ndarray, bbox: BoundingBox, num_bins: int = 36) -> list[float]:
        """Calculate a simple hue histogram (placeholder implementation)"""
        # Simplified histogram - in real implementation this would extract ROI and calculate hue
        return [1.0 / num_bins] * num_bins

    def get_track_count(self) -> int:
        """Get the number of unique tracks"""
        return len(self.track_inputs)

    def get_tracks_dict(self) -> TrackerInput:
        """Get all tracks as a dictionary"""
        return dict(self.track_inputs)

    def process_tracks(self, original_video_path) -> TrackerInput:
        """Process collected tracks and return processed track data for video generation"""
        logger = logging.getLogger(__name__)

        if not self.track_inputs:
            logger.warning('No tracks available for processing')
            return {}

        # Get video properties for track processing
        video_properties = get_video_properties(original_video_path)

        # Process tracks using the track processing module
        processed_tracks = track_processing.process_tracks(self.track_inputs, video_properties)

        if not processed_tracks:
            logger.warning('No valid tracks found for video generation')
            return {}

        # Log track statistics
        logger.info(f'After processing: {len(processed_tracks)} tracks remaining')
        for track_id, track_data in processed_tracks.items():
            duration_frames = track_data[-1].frame_idx - track_data[0].frame_idx
            duration_seconds = duration_frames / video_properties.fps
            frame_percentage = duration_frames / video_properties.total_frames
            logger.info(
                f'  Track {track_id}: {len(track_data)} detections, {duration_seconds:.1f}s ({frame_percentage * 100:.1f}%)'
            )

        # Generate individual videos using the video splicing module
        return processed_tracks
