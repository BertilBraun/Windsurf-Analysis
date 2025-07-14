"""
Orchestrator functions for coordinating track processing and video generation.

This module provides high-level functions that coordinate between track processing
and video splicing modules to provide a simple interface for the complete pipeline.
"""

import os
import numpy as np
import cv2
from collections import defaultdict

from detector import Detection
from video_io import get_video_properties
import track_processing
import video_splicing


class SurferTracker:
    def __init__(self):
        self.track_inputs: track_processing.TrackerInput = defaultdict(list)

    def add_detection(self, frame_idx: int, detection: Detection, frame: np.ndarray):
        # Calculate hue histogram from bounding box region
        hue_histogram = self._calculate_hue_histogram(frame, detection.bbox)

        self.track_inputs[detection.track_id].append(
            track_processing.Track(frame_idx, detection.bbox.copy(), detection.confidence, hue_histogram)
        )

    def _calculate_hue_histogram(self, frame: np.ndarray, bbox, num_bins: int = 36) -> list[float]:
        """Calculate a normalized hue histogram from the bounding box region of the frame.

        Args:
            frame: BGR image frame from OpenCV
            bbox: BoundingBox object with x1, y1, x2, y2 coordinates
            num_bins: Number of bins for the histogram (default: 36 for 10° per bin)

        Returns:
            Normalized hue histogram as a list of floats (frequencies sum to 1.0)
        """
        # Extract the region of interest (ROI) from the frame
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(bbox.x1, w - 1))
        y1 = max(0, min(bbox.y1, h - 1))
        x2 = max(x1 + 1, min(bbox.x2, w))
        y2 = max(y1 + 1, min(bbox.y2, h))

        roi = frame[y1:y2, x1:x2]

        # Handle edge case where ROI is empty
        if roi.size == 0:
            # Return uniform distribution as fallback
            return [1.0 / num_bins] * num_bins

        # Convert BGR to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Extract hue channel (0-179 in OpenCV)
        hue_channel = hsv_roi[:, :, 0]

        # Create histogram with specified number of bins
        # OpenCV hue range is 0-179, so we need to scale our bins accordingly
        hist, _ = np.histogram(hue_channel.flatten(), bins=num_bins, range=(0, 180))

        # Normalize by total pixel count to get frequencies
        total_pixels = hue_channel.size
        if total_pixels == 0:
            # Return uniform distribution as fallback
            return [1.0 / num_bins] * num_bins

        normalized_hist = hist.astype(float) / total_pixels

        return normalized_hist.tolist()

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
        processed_tracks = track_processing.process_tracks(self.track_inputs, video_properties)

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
