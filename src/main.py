#!/usr/bin/env python3

import glob
import torch
import argparse
import traceback


from settings import STANDARD_OUTPUT_DIR
from windsurf_video_processor import WindsurfingVideoProcessor


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

    processor.finalize()


if __name__ == '__main__':
    main()
