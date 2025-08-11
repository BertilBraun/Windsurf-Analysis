#!/usr/bin/env python3

import glob
import logging
import torch
import argparse
import traceback
from pathlib import Path
from itertools import chain

from concurrent.futures import ProcessPoolExecutor, as_completed

from util.helpers import setup_logging

from settings import NUM_PARALLEL_VIDEO_WORKERS, STANDARD_OUTPUT_DIR
from windsurf_video_processor import WindsurfingVideoProcessor


def main():
    parser = argparse.ArgumentParser(description='Windsurfing Video Analysis Tool')
    parser.add_argument(
        'input_pattern', nargs='+', help='Path pattern for input video files (e.g., "videos/*.mp4" or single file)'
    )
    parser.add_argument(
        '--output-dir',
        default=STANDARD_OUTPUT_DIR,
        help='Directory for individual surfer videos (default: individual_surfers)',
    )
    parser.add_argument('--generate-videos', action='store_true', help='Generate individual videos')
    parser.add_argument('--draw-annotations', action='store_true', help='Draw annotations on the video')
    parser.add_argument('--debug-views', action='store_true', help='Output debug views of the video processing steps')
    parser.add_argument('--stabilize', action='store_true', help='Stabilize the video')
    parser.add_argument('--parallel-workers', type=int, default=1, help='Number of parallel workers to use')

    args = parser.parse_args()

    output_dir_path = Path(args.output_dir)
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

    _log_detection_settings(logger)

    parallel_workers = args.parallel_workers
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = []
        for worker_id in range(parallel_workers):
            indices_to_process = [i for i in range(len(video_files)) if i % parallel_workers == worker_id]
            futures.append(
                executor.submit(
                    _process_videos,
                    video_files,
                    indices_to_process,
                    output_dir_path,
                    args.draw_annotations,
                    args.generate_videos,
                    args.debug_views,
                    args.stabilize,
                )
            )

        for future in as_completed(futures):
            future.result()


def _log_detection_settings(logger: logging.Logger):
    import settings

    settings_str = '\n'.join(
        f'{k}: {v}' for k, v in settings.__dict__.items() if not k.startswith('__') and not callable(v) and k.isupper()
    )
    logger.info(f'Detection settings: \n{settings_str}')


def _process_videos(
    video_files: list[str],
    indices_to_process: list[int],
    output_dir: Path | None,
    draw_annotations: bool,
    generate_videos: bool,
    debug_views: bool,
    stabilize: bool,
):
    logger = setup_logging(output_dir)

    processor = WindsurfingVideoProcessor(
        draw_annotations=draw_annotations,
        output_dir=str(output_dir) if output_dir else STANDARD_OUTPUT_DIR,
        generate_videos=generate_videos,
        debug_views=debug_views,
        parallel_workers=NUM_PARALLEL_VIDEO_WORKERS,
        stabilize=stabilize,
    )

    for i, video_file in enumerate(video_files):
        if i not in indices_to_process:
            continue

        logger.info(f'Processing video {i}/{len(video_files)}: {video_file}')
        try:
            processor.process_video(Path(video_file))
            logger.info(f'✓ Completed processing: {video_file}')
        except Exception as e:
            logger.error(f'✗ Error processing {video_file}: {e}')
            logger.error(traceback.format_exc())

    processor.finalize()


if __name__ == '__main__':
    main()
