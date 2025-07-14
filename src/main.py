#!/usr/bin/env python3

import os
import argparse
import glob

import torch
from tqdm import tqdm

from video_io import get_video_properties
from detector import SurferDetector
from surfer_tracker import SurferTracker, TrackerInput

from settings import STANDARD_OUTPUT_DIR


class WindsurfingVideoProcessor:
    def __init__(self):
        self.detector = SurferDetector()

    def process_video(self, input_path: os.PathLike, surfer_output_dir: os.PathLike | str):
        """Main video processing pipeline with batched YOLO inference"""

        surfer_tracker_input = TrackerInput()

        props = get_video_properties(input_path)
        print(f'Processing video: {props.width}x{props.height}, {props.fps} FPS, {props.total_frames} frames')

        for frame_index, detections in tqdm(
            enumerate(self.detector.detect_and_track_video(input_path)),
            total=props.total_frames,
            desc='Processing video',
        ):
            for detection in detections:
                if detection.track_id is not None:
                    surfer_tracker_input.add_detection(
                        frame_index, detection.track_id, detection.bbox, detection.confidence
                    )

        SurferTracker().process_tracks(input_path, surfer_tracker_input, surfer_output_dir)


def main():
    parser = argparse.ArgumentParser(description='Windsurfing Video Analysis Tool')
    parser.add_argument(
        'input_pattern', help='Path pattern for input video files (e.g., "videos/*.mp4" or single file)'
    )
    parser.add_argument('--output-dir', help='Directory for individual surfer videos (default: individual_surfers)')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('=' * 80)
        print('WARNING: CUDA is not available. This will be slow.')
        print('=' * 80)

    # Expand glob pattern to find matching video files
    video_files = glob.glob(args.input_pattern)

    if not video_files:
        print(f'No video files found matching pattern: {args.input_pattern}')
        return

    # Sort files for consistent processing order
    video_files.sort()

    print(f'Found {len(video_files)} video file(s) to process:')
    for video_file in video_files:
        print(f'  - {video_file}')
    print()

    processor = WindsurfingVideoProcessor()

    for i, video_file in enumerate(video_files, 1):
        print(f'Processing video {i}/{len(video_files)}: {video_file}')
        try:
            processor.process_video(video_file, args.output_dir or STANDARD_OUTPUT_DIR)
            print(f'✓ Completed processing: {video_file}')
        except Exception as e:
            print(f'✗ Error processing {video_file}: {e}')
        print('-' * 60)


if __name__ == '__main__':
    main()
