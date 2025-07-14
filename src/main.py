#!/usr/bin/env python3

from collections import defaultdict
import os
import argparse
import glob
from pathlib import Path

import torch
from tqdm import tqdm

from video_io import VideoReader, VideoWriter, get_video_properties
from detector import Detection, SurferDetector
from surfer_tracker import SurferTracker
from annotation_drawer import AnnotationDrawer

from settings import STANDARD_OUTPUT_DIR


class WindsurfingVideoProcessor:
    def __init__(self):
        self.detector = SurferDetector()

    def process_video(self, input_path: os.PathLike, output_dir: os.PathLike | str):
        """Main video processing pipeline with batched YOLO inference"""

        surfer_tracker = SurferTracker()

        props = get_video_properties(input_path)
        print(f'Processing video: {props.width}x{props.height}, {props.fps} FPS, {props.total_frames} frames')

        all_detections: dict[int, list[Detection]] = defaultdict(list)

        for frame_index, detections in tqdm(
            enumerate(self.detector.detect_and_track_video(input_path)),
            total=props.total_frames,
            desc='Processing video',
        ):
            for detection in detections:
                if detection.track_id is not None:
                    surfer_tracker.add_detection(frame_index, detection.track_id, detection.bbox, detection.confidence)

            all_detections[frame_index] = detections

        surfer_tracker.process_tracks(input_path, output_dir)
        self.generate_annotated_video(input_path, all_detections, output_dir)

    def generate_annotated_video(
        self, input_path: os.PathLike, detections: dict[int, list[Detection]], output_dir: os.PathLike | str
    ):
        annotation_drawer = AnnotationDrawer()

        output_path = Path(output_dir) / f'{Path(input_path).stem}+00_annotated.mp4'
        with VideoReader(input_path) as reader:
            video_props = reader.get_properties()
            with VideoWriter(output_path, video_props.width, video_props.height, video_props.fps) as writer:
                for frame_index, frame in reader.read_frames():
                    writer.write_frame(
                        annotation_drawer.draw_detections_with_trails(frame, detections[frame_index] or [])
                    )


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
            import traceback

            print(traceback.format_exc())
        print('-' * 60)


if __name__ == '__main__':
    main()
