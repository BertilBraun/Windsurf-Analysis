"""
Video splicing functions for extracting and generating individual tracked object videos.

This module provides pure functions for video processing operations without
maintaining any state, making it easier to test and reason about.
"""

import json
import os
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

from common_types import Point
from track_processing import Track, TrackId, TrackerInput
from video_io import VideoReader, VideoWriter, get_video_properties


def _extract_centered_slice(frame: np.ndarray, center: Point, slice_size: tuple[int, int]) -> np.ndarray:
    """Extract a fixed-size slice centered on a point with black padding if needed.

    Args:
        frame: Input frame
        center: Center point for extraction
        slice_size: (width, height) of the output slice

    Returns:
        Fixed-size slice with black padding if needed
    """
    frame_height, frame_width = frame.shape[:2]
    slice_width, slice_height = slice_size

    # Calculate slice boundaries centered on bbox center
    slice_x1 = int(center.x - slice_width // 2)
    slice_y1 = int(center.y - slice_height // 2)
    slice_x2 = slice_x1 + slice_width
    slice_y2 = slice_y1 + slice_height

    # Create output slice with black background
    output_slice = np.zeros((slice_height, slice_width, 3), dtype=np.uint8)

    # Calculate intersection with frame bounds
    frame_x1 = max(0, slice_x1)
    frame_y1 = max(0, slice_y1)
    frame_x2 = min(frame_width, slice_x2)
    frame_y2 = min(frame_height, slice_y2)

    # Calculate corresponding positions in output slice
    out_x1 = frame_x1 - slice_x1
    out_y1 = frame_y1 - slice_y1
    out_x2 = out_x1 + (frame_x2 - frame_x1)
    out_y2 = out_y1 + (frame_y2 - frame_y1)

    # Copy the valid region from frame to output slice
    if frame_x2 > frame_x1 and frame_y2 > frame_y1:
        output_slice[out_y1:out_y2, out_x1:out_x2] = frame[frame_y1:frame_y2, frame_x1:frame_x2]

    return output_slice


def _calculate_crop_size(track_data: list[Track], min_width: int = 400, min_height: int = 600) -> tuple[int, int]:
    """Calculate optimal crop size for a track based on average bbox dimensions.

    Args:
        track_data: List of track detections
        min_width: Minimum width for the crop
        min_height: Minimum height for the crop

    Returns:
        Tuple of (width, height) for the crop
    """
    if not track_data:
        return min_width, min_height

    # Calculate average bbox size for this track
    total_width = sum(detection.bbox.width for detection in track_data)
    total_height = sum(detection.bbox.height for detection in track_data)

    avg_width = total_width / len(track_data)
    avg_height = total_height / len(track_data)
    avg_width = max(detection.bbox.width for detection in track_data)
    avg_height = max(detection.bbox.height for detection in track_data)

    # Create slice size with generous context
    context_factor = 2.0
    slice_width = int(avg_width * context_factor)
    slice_height = int(avg_height * context_factor)

    # Ensure minimum reasonable size
    slice_width = max(slice_width, min_width)
    slice_height = max(slice_height, min_height)

    # Scale to 1000-2000px range to reduce artifacts
    current_max = max(slice_width, slice_height)
    target_size = 1500  # Target for larger dimension

    if current_max < 1000:
        # Scale up if too small
        scale_factor = target_size / current_max
        slice_width = int(slice_width * scale_factor)
        slice_height = int(slice_height * scale_factor)

    # Round up to even numbers for video encoding compatibility
    slice_width = int(np.ceil(slice_width / 2) * 2)
    slice_height = int(np.ceil(slice_height / 2) * 2)

    return slice_width, slice_height


def _find_detection_at_frame(track_data: list[Track], frame_idx: int) -> Track | None:
    """Find the detection at a specific frame index in track data.

    Args:
        track_data: List of track detections
        frame_idx: Frame index to search for

    Returns:
        Detection at the frame index, or None if not found
    """
    for detection in track_data:
        if detection.frame_idx == frame_idx:
            return detection
    return None


def generate_individual_videos(
    tracks: TrackerInput, original_video_path: os.PathLike | str, output_dir: os.PathLike | str
) -> list[os.PathLike]:
    """Generate individual MP4 videos for each tracked person with centered, fixed-size crops.

    Args:
        tracks: Dictionary of processed track data
        original_video_path: Path to original high-resolution video
        output_dir: Directory to save individual videos
    """
    logger = logging.getLogger(__name__)

    if not tracks:
        logger.warning('No tracks found for individual video generation')
        return []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get input filename without extension for output naming
    input_name = Path(original_video_path).stem

    # Read original video properties
    video_properties = get_video_properties(original_video_path)
    total_frames = video_properties.total_frames

    logger.info(f'Generating individual videos for {len(tracks)} tracks...')

    # Pre-calculate crop sizes and create writers
    writers: dict[TrackId, VideoWriter] = {}
    crop_sizes: dict[TrackId, tuple[int, int]] = {}

    for person_number, track_data in tracks.items():
        # Calculate optimal crop size for this track
        slice_width, slice_height = _calculate_crop_size(track_data)
        crop_sizes[person_number] = (slice_width, slice_height)

        logger.info(f'Track {person_number}: slice size {slice_width}x{slice_height} pixels')

        # Create video writer with sequential numbering
        output_path = Path(output_dir) / f'{input_name}+{person_number:02d}.mp4'
        writer = VideoWriter(output_path, slice_width, slice_height, video_properties.fps)
        writer.start_writing()
        writers[person_number] = writer

        start_time_path = Path(output_dir) / f'{input_name}+{person_number:02d}.start_time.json'
        start_time_path.write_text(json.dumps({'start_time': track_data[0].frame_idx / video_properties.fps}))

    # Process video frame by frame with progress bar
    with VideoReader(original_video_path) as reader:
        for frame_idx, frame in tqdm(reader.read_frames(), total=total_frames, desc='Writing individual videos'):
            # Process each track for this frame
            for track_id, track_data in tracks.items():
                # Find detection for this frame
                detection = _find_detection_at_frame(track_data, frame_idx)

                if detection is not None:
                    # Extract fixed-size slice centered on bbox
                    target_size = crop_sizes[track_id]
                    cropped_frame = _extract_centered_slice(frame, detection.bbox.center, target_size)
                    writers[track_id].write_frame(cropped_frame)

    # Clean up writers
    for writer in writers.values():
        writer.finish_writing()

    logger.info(f'Individual videos saved to: {output_dir}')

    return [Path(writer.output_path) for writer in writers.values()]
