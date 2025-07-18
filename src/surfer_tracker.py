"""
Surfer tracking module for aggregating detections across frames.

This module handles the collection and organization of YOLO detections
into coherent tracks representing individual surfers.
"""

import logging
from collections import defaultdict
import os

import tracking
import track_processing
from common_types import Detection, FrameIndex, TrackerInput
from video_io import get_video_properties


class SurferTracker:
    """Aggregates YOLO detections into tracks for individual surfers"""

    def __init__(self):
        self.detections: dict[FrameIndex, list[Detection]] = defaultdict(list)

    def add_detection(self, frame_idx: FrameIndex, detection: Detection):
        self.detections[frame_idx].append(detection)

    def process_tracks(self, original_video_path: os.PathLike | str) -> TrackerInput:
        """Process collected tracks and return processed track data for video generation"""
        logger = logging.getLogger(__name__)

        if not self.detections:
            logger.warning('No tracks available for processing')
            return {}

        # Get video properties for track processing
        video_properties = get_video_properties(original_video_path)

        processed_tracks = tracking.process_detections(self.detections, video_properties)

        # Process tracks using the track processing module
        processed_tracks = track_processing.process_tracks(processed_tracks, video_properties)

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

        return processed_tracks
