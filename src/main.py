#!/usr/bin/env python3

import os
import glob
import torch
import argparse
import traceback

from tqdm import tqdm
from pathlib import Path


from video_io import VideoReader, VideoWriter, get_video_properties
from detector import SurferDetector
from surfer_tracker import SurferTracker
from annotation_drawer import Annotation, AnnotationDrawer
from stabilize import stabilize_ffmpeg

from settings import STANDARD_OUTPUT_DIR
from common_types import TrackerInput
from worker_pool import WorkerPool


def _stabilize_worker_function(args: tuple[os.PathLike | str, os.PathLike | str]) -> None:
    input_file, output_file = args
    stabilize_ffmpeg(input_file, output_file)


def _write_annotated_video_worker_function(args: tuple[TrackerInput, os.PathLike, os.PathLike | str]) -> None:
    tracks, input_path, output_dir = args
    annotation_drawer = AnnotationDrawer()

    annotated_video_path = Path(output_dir) / f'{Path(input_path).stem}+00_annotated.mp4'

    with VideoReader(input_path) as reader:
        video_props = reader.get_properties()
        with VideoWriter(annotated_video_path, video_props.width, video_props.height, video_props.fps) as writer:
            for frame_index, frame in tqdm(
                reader.read_frames(), total=video_props.total_frames, desc='Drawing annotations'
            ):
                annotations = [
                    Annotation(track_id, track.bbox, track.confidence)
                    for track_id, tracks in tracks.items()
                    for track in tracks
                    if track.frame_idx == frame_index
                ]

                writer.write_frame(annotation_drawer.draw_detections_with_trails(frame, annotations))


class WindsurfingVideoProcessor:
    def __init__(self, draw_annotations: bool, output_dir: os.PathLike | str):
        self.detector = SurferDetector()
        self.stabilizer = WorkerPool(_stabilize_worker_function, num_workers=1)
        self.annotated_video_writer = WorkerPool(_write_annotated_video_worker_function, num_workers=1)
        self.draw_annotations = draw_annotations
        self.output_dir = output_dir

    def process_video(self, input_path: os.PathLike):
        """Main video processing pipeline with batched YOLO inference"""

        surfer_tracker = SurferTracker()

        props = get_video_properties(input_path)
        print(f'Processing video: {props.width}x{props.height}, {props.fps} FPS, {props.total_frames} frames')

        for frame_index, frame, detections in self.detector.detect_and_track_video(input_path):
            for detection in detections:
                if detection.track_id is not None:
                    surfer_tracker.add_detection(frame_index, detection, frame)

        processed_tracks, individual_videos = surfer_tracker.process_tracks(input_path, self.output_dir)
        for individual_video in individual_videos:
            self.stabilizer.submit((individual_video, individual_video))

        if self.draw_annotations:
            all_tracks = {track_id: tracks for track_id, tracks in processed_tracks.items()}
            all_tracks.update(surfer_tracker.track_inputs)
            # YOLO tracks all_tracks = {track_id: tracks for track_id, tracks in surfer_tracker.track_inputs.items()}

            self.annotated_video_writer.submit((all_tracks, input_path, self.output_dir))

    def finalize(self):
        self.stabilizer.stop()
        self.annotated_video_writer.stop()


def main():
    parser = argparse.ArgumentParser(description='Windsurfing Video Analysis Tool')
    parser.add_argument(
        'input_pattern', help='Path pattern for input video files (e.g., "videos/*.mp4" or single file)'
    )
    parser.add_argument('--output-dir', help='Directory for individual surfer videos (default: individual_surfers)')
    parser.add_argument('--draw-annotations', action='store_true', help='Draw annotations on the video')

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

    processor = WindsurfingVideoProcessor(
        draw_annotations=args.draw_annotations,
        output_dir=args.output_dir or STANDARD_OUTPUT_DIR,
    )

    for i, video_file in enumerate(video_files, 1):
        print(f'Processing video {i}/{len(video_files)}: {video_file}')
        try:
            processor.process_video(video_file)
            print(f'✓ Completed processing: {video_file}')
        except Exception as e:
            print(f'✗ Error processing {video_file}: {e}')
            print(traceback.format_exc())

        print('-' * 60)

    processor.finalize()


if __name__ == '__main__':
    main()
