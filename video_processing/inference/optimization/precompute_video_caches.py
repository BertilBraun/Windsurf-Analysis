from __future__ import annotations

import sys
import glob
import argparse
from pathlib import Path


# Make project importable when run as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from video_processing.inference.src.settings import YOLO_MODEL_PATH
from video_processing.inference.src.tracking.detector import SurferDetector
from video_processing.inference.src.util.video_io import get_video_properties
from video_processing.inference.src.visualization.stabilize import compute_stabilization_transforms_gmc


def _expand_video_paths(patterns: list[str]) -> list[Path]:
    video_paths: list[Path] = []
    for pat in patterns:
        expanded = [Path(p) for p in glob.glob(pat)]
        if not expanded:
            p = Path(pat)
            if p.exists():
                expanded = [p]
        video_paths.extend(expanded)
    return sorted({p.resolve() for p in video_paths if p.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}})


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Precompute heavy caches (YOLO/ReID + GMC stabilization) for a batch of videos.'
    )
    parser.add_argument('videos', type=str, nargs='+', help='Path(s) or glob pattern(s) to input video(s)')
    parser.add_argument('--skip-detections', action='store_true', help='Skip YOLO/ReID detection+embedding cache')
    parser.add_argument('--skip-stabilization', action='store_true', help='Skip GMC stabilization transform cache')
    args = parser.parse_args()

    videos = _expand_video_paths(list(args.videos))
    if not videos:
        print('No input videos found for given patterns.')
        return

    detector = None if args.skip_detections else SurferDetector(yolo_model_path=YOLO_MODEL_PATH)

    for idx, video in enumerate(videos, start=1):
        try:
            print(f'[{idx}/{len(videos)}] {video}')
            _ = get_video_properties(video.as_posix())

            if detector is not None:
                # Warms both YOLO raw detections and ReID embedding caches.
                dets = detector.run_object_detection_on_video(video.as_posix())
                print(f'  detections: {len(dets)}')

            if not args.skip_stabilization:
                transforms = compute_stabilization_transforms_gmc(video.as_posix())
                print(f'  transforms: {len(transforms)}')
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f'  ERROR: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()

