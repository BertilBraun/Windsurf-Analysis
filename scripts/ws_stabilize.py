#!/usr/bin/env python3

import shutil
import logging
import argparse

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from util import stabilize_ffmpeg, setup_logging, get_tmp_file, mby_notify


def stabilize_one(input_file: Path, output_file: Path, force: bool):
    if not input_file.exists():
        logging.error(f'Input file {input_file} does not exist.')
        return (input_file, False, 'Input file does not exist.')

    if output_file.exists() and not force:
        logging.warning(f'Output file {output_file} already exists. Use --force to overwrite.')
        return (input_file, False, 'Output exists')

    # Worker process: don't setup logging here.
    temp_output = get_tmp_file(suffix='_stabilized.mp4')
    try:
        success = stabilize_ffmpeg(input_file, Path(temp_output))

        if success:
            shutil.move(temp_output, output_file)
            return (input_file, True, str(output_file))
        else:
            try:
                Path(temp_output).unlink()
            except Exception:
                pass
            return (input_file, False, 'Stabilization failed')
    finally:
        try:
            Path(temp_output).unlink()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='Stabilize video files using FFmpeg (parallelized)')
    parser.add_argument('input_files', nargs='+', help='Path(s) to the input video file(s)')
    parser.add_argument(
        '-o',
        '--output',
        help='Output file (single input) or directory (multiple inputs). If omitted, defaults to current directory.',
    )
    parser.add_argument('-f', '--force', action='store_true', help='Force stabilization even if the output file exists')
    parser.add_argument('--no-suffix', action='store_true', help="Do not append '_stabilized' to the output file name")
    parser.add_argument('-j', '--jobs', type=int, default=2, help='Number of parallel workers')
    parser.add_argument('--log-file', help='Log to a file instead of stdout', type=str, default=None)
    args = parser.parse_args()

    setup_logging(notify_on_error=True, log_file=args.log_file)
    input_files = [Path(f).resolve() for f in args.input_files]
    output_arg = Path(args.output).resolve() if args.output else None
    multi = len(input_files) > 1

    # Output directory logic
    outputs = []
    if multi or (output_arg and output_arg.is_dir()):
        output_dir = output_arg if output_arg else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        for input_file in input_files:
            if args.no_suffix:
                basename = input_file.name
            else:
                basename = f'{input_file.stem}_stabilized{input_file.suffix}'
            output_file = output_dir / basename
            outputs.append((input_file, output_file, args.force))
    else:
        input_file = input_files[0]
        if output_arg:
            if output_arg.exists() and output_arg.is_dir():
                if args.no_suffix:
                    basename = input_file.name
                else:
                    basename = f'{input_file.stem}_stabilized{input_file.suffix}'
                output_file = output_arg / basename
            else:
                output_file = output_arg
        else:
            if args.no_suffix:
                output_file = input_file.with_name(input_file.name)
            else:
                output_file = input_file.with_name(f'{input_file.stem}_stabilized{input_file.suffix}')
        outputs.append((input_file, output_file, args.force))

    # Parallelize with logging-aware progress bar
    total = len(outputs)
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        future_to_task = {executor.submit(stabilize_one, *task): task for task in outputs}
        for future in tqdm(as_completed(future_to_task), total=total, desc='Stabilizing'):
            result = future.result()
            input_file, success, info = result
            if success:
                logging.info(f'Stabilized: {input_file} -> {info}')
            else:
                logging.error(f'Failed: {input_file} | {info}')
            results.append(result)

    failed = [f for f in results if not f[1]]
    if failed:
        logging.error(f'{len(failed)} file(s) failed to stabilize:')
        for f in failed:
            logging.error(f'  {f[0]}: {f[2]}')
    else:
        mby_notify('Stabilization Complete', 'All files stabilized successfully.')
        logging.info('All files stabilized successfully.')


if __name__ == '__main__':
    main()
