#!/usr/bin/env python3

import glob
import torch
import argparse
import traceback
import logging
from pathlib import Path
from itertools import chain


from settings import STANDARD_OUTPUT_DIR
from windsurf_video_processor import WindsurfingVideoProcessor


def setup_logging(output_dir: Path | None = None):
    """Configure logging for the windsurfing video analysis tool."""
    if output_dir is None:
        output_dir = Path(STANDARD_OUTPUT_DIR)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / 'windsurf_analysis.log')],
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Windsurfing Video Analysis Tool')
    parser.add_argument(
        'input_pattern', nargs='+', help='Path pattern for input video files (e.g., "videos/*.mp4" or single file)'
    )
    parser.add_argument('--output-dir', help='Directory for individual surfer videos (default: individual_surfers)')
    parser.add_argument('--draw-annotations', action='store_true', help='Draw annotations on the video')
    parser.add_argument('--dry-run', action='store_true', help='Run without rendering individual videos (for testing purposes)')

    args = parser.parse_args()

    output_dir_path = Path(args.output_dir) if args.output_dir else None

    if output_dir_path:
        output_dir_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir_path)

    if not torch.cuda.is_available():
        logger.warning('=' * 80)
        logger.warning('WARNING: CUDA is not available. This will be slow.')
        logger.warning('=' * 80)

    # Expand glob pattern to find matching video files
    video_files = list(chain(*(glob.glob(p) for p in args.input_pattern)))

    if not video_files:
        logger.error(f'No video files found matching pattern: {args.input_pattern}')
        return

    # Sort files for consistent processing order
    video_files.sort()

    logger.info(f'Found {len(video_files)} video file(s) to process:')
    for video_file in video_files:
        logger.info(f'  - {video_file}')

    processor = WindsurfingVideoProcessor(
        draw_annotations=args.draw_annotations,
        output_dir=args.output_dir or STANDARD_OUTPUT_DIR,
        dry_run=args.dry_run,
    )

    for i, video_file in enumerate(video_files, 1):
        logger.info(f'Processing video {i}/{len(video_files)}: {video_file}')
        try:
            processor.process_video(video_file)
            logger.info(f'✓ Completed processing: {video_file}')
        except Exception as e:
            logger.error(f'✗ Error processing {video_file}: {e}')
            logger.error(traceback.format_exc())

    processor.finalize()


if __name__ == '__main__':
    main()
