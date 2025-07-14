"""
Orchestrator functions for coordinating track processing and video generation.

This module provides high-level functions that coordinate between track processing
and video splicing modules to provide a simple interface for the complete pipeline.
"""

import os
import numpy as np
from collections import defaultdict

from detector import Detection
from video_io import get_video_properties
import track_processing
import video_splicing


class SurferTracker:
    def __init__(self):
        self.track_inputs: track_processing.TrackerInput = defaultdict(list)

    def add_detection(self, frame_idx: int, detection: Detection, frame: np.ndarray):
        self.track_inputs[detection.track_id].append(
            track_processing.Track(frame_idx, detection.bbox.copy(), detection.confidence)
        )

    def process_tracks(
        self, original_video_path: os.PathLike, output_dir: os.PathLike | str
    ) -> tuple[track_processing.TrackerInput, list[os.PathLike | str]]:
        """Complete pipeline: process tracks and generate individual videos.

        This is the main entry point that coordinates track processing and video generation.

        Args:
            original_video_path: Path to original high-resolution video
            output_dir: Directory to save individual videos
        """
        # Get video properties for track processing
        video_properties = get_video_properties(original_video_path)

        # Process tracks using the track processing module
        processed_tracks = track_processing.process_tracks(self.track_inputs, video_properties.total_frames)

        if not processed_tracks:
            print('No valid tracks found for video generation')
            return {}, []

        # Print track statistics
        print(f'After processing: {len(processed_tracks)} tracks remaining')
        for track_id, track_data in processed_tracks.items():
            duration_frames = track_data[-1].frame_idx - track_data[0].frame_idx
            duration_seconds = duration_frames / video_properties.fps
            frame_percentage = duration_frames / video_properties.total_frames
            print(
                f'  Track {track_id}: {len(track_data)} detections, {duration_seconds:.1f}s ({frame_percentage * 100:.1f}%)'
            )

        # Generate individual videos using the video splicing module
        return processed_tracks, video_splicing.generate_individual_videos(
            processed_tracks, original_video_path, output_dir
        )
