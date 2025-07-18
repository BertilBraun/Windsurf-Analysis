"""
Surfer tracking module for aggregating detections across frames.

This module handles the collection and organization of YOLO detections
into coherent tracks representing individual surfers.
"""

import os
import logging

import tracking
import track_processing
from vidstab import VidStab

from common_types import Detection, Track
from video_io import get_video_properties


def process_detections_into_tracks(
    original_video_path: os.PathLike | str, detections: list[Detection], stabilizer: VidStab
) -> list[Track]:
    """Process collected tracks and return processed track data for video generation"""
    logger = logging.getLogger(__name__)

    if not detections:
        logger.warning('No tracks available for processing')
        return []

    # Get video properties for track processing
    video_properties = get_video_properties(original_video_path)

    processed_tracks = tracking.process_detections(detections, video_properties, stabilizer)

    # Process tracks using the track processing module
    processed_tracks = track_processing.tracks_filtering_smoothing_relabeling(processed_tracks, video_properties)

    if not processed_tracks:
        logger.warning('No valid tracks found for video generation')
        return []

    # Log track statistics
    logger.info(f'After processing: {len(processed_tracks)} tracks remaining')
    for track in processed_tracks:
        duration_frames = track.sorted_detections[-1].frame_idx - track.sorted_detections[0].frame_idx
        duration_seconds = duration_frames / video_properties.fps
        frame_percentage = duration_frames / video_properties.total_frames
        logger.info(
            f'  Track {track.track_id}: {len(track.sorted_detections)} detections, {duration_seconds:.1f}s ({frame_percentage * 100:.1f}%)'
        )

    return processed_tracks
