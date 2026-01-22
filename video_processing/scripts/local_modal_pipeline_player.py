from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


def _add_video_processing_to_path() -> None:
    import sys

    video_processing_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(video_processing_dir))


_add_video_processing_to_path()

from inference.src.common_types import BoundingBox, Detection, Keypoint, RenderableTrack, Track
from inference.src.orientation_fixer import OrientationFixer
from inference.src.player.core.player_state import DetectionLite, Metadata, TrackLite, VideoProperties
from inference.src.settings import REID_MODEL_TYPE, YOLO_MODEL_PATH
from inference.src.tracking.detector import EmbeddingExtractor, ObjectDetector, RawDetection
from inference.src.tracking.ilp_tracker import ILPTracker
from inference.src.tracking.iterative_ilp_tracker import IterativeILPTracker
from inference.src.tracking.preprocessing.preprocessor import TrackPreProcessor
from inference.src.tracking.track_processing import TrackPostProcessing, prepare_renderable_tracks
from inference.src.util.timing import timeit
from inference.src.util.video_io import VideoReader, get_video_properties
from inference.src.visualization.stabilize import (
    STABLE_GFTT_BLOCK_SIZE,
    STABLE_GFTT_MAX_CORNERS,
    STABLE_GFTT_MIN_DISTANCE,
    STABLE_GFTT_QUALITY_LEVEL,
    STABLE_SMOOTHING_WINDOW,
    Transform,
    compute_stabilization_transforms,
    compute_stabilization_transforms_masked_vidstab,
    compute_stabilization_transforms_gmc,
    gmc_transform_from_frame,
)
from inference.src.motion.gmc import GMC
from inference.src.util.cache import cache_to_file


StabilizerName = Literal['masked_vidstab', 'gmc', 'vidstab', 'none']


@dataclass(frozen=True)
class LocalRunResult:
    output_dir: Path
    upright_video_path: Path
    metadata_path: Path
    stabilization_transforms_path: Path | None
    raw_motion_transforms_path: Path | None
    dominant_orientation: int
    transforms: list[Transform]
    tracks: list[Track]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Hacky local end-to-end runner that mimics the Modal pipeline:\n'
            '  orientation -> YOLO detection -> stabilization transforms -> embeddings -> tracking -> post-processing\n'
            'It writes a `.tracks.pkl` metadata file and optionally launches the local Qt player.'
        )
    )
    p.add_argument('--input-video', type=str, required=True)
    p.add_argument('--output-dir', type=str, default='tmp/local_modal_run')

    p.add_argument('--skip-orientation', action='store_true')
    p.add_argument(
        '--orientation-model-path',
        type=str,
        default=None,
        help='Defaults to `video_processing/inference/weights/orientation_fixer/best.pt`.',
    )

    p.add_argument(
        '--yolo-model-path', type=str, default=None, help='Defaults to `inference.src.settings.YOLO_MODEL_PATH`.'
    )
    p.add_argument(
        '--limit-frames', type=int, default=None, help='Optionally limit processed frames for faster iteration.'
    )
    p.add_argument(
        '--use-detector-crops',
        action='store_true',
        help='Use crops produced by YOLO detection pass (faster, but less faithful to Modal tracking pipeline).',
    )

    p.add_argument(
        '--stabilizer',
        type=str,
        default='masked_vidstab',
        choices=['masked_vidstab', 'gmc', 'vidstab', 'none'],
        help='Which per-frame camera motion estimator to use.',
    )
    p.add_argument(
        '--mask-margin-px', type=int, default=20, help='Padding around detected bboxes when masking keypoints.'
    )
    p.add_argument(
        '--processing-max-dim',
        type=float,
        default=None,
        help='Downscale max dimension for masked_vidstab estimator (default: half of input max dimension).',
    )
    p.add_argument('--smoothing-window', type=int, default=int(STABLE_SMOOTHING_WINDOW))
    p.add_argument('--masked-max-corners', type=int, default=int(STABLE_GFTT_MAX_CORNERS))
    p.add_argument('--masked-quality-level', type=float, default=float(STABLE_GFTT_QUALITY_LEVEL))
    p.add_argument('--masked-min-distance', type=float, default=float(STABLE_GFTT_MIN_DISTANCE))
    p.add_argument('--masked-block-size', type=int, default=int(STABLE_GFTT_BLOCK_SIZE))
    p.add_argument('--gmc-downscale', type=int, default=2, help='GMC downscale factor.')
    p.add_argument(
        '--masked-vidstab-debug-dir',
        type=str,
        default=None,
        help='If set and --stabilizer=masked_vidstab, write mask/keypoint debug frames into this directory.',
    )
    p.add_argument(
        '--masked-vidstab-debug-every-n',
        type=int,
        default=1,
        help='Write every Nth debug frame (only when --masked-vidstab-debug-dir is set).',
    )

    p.add_argument('--no-player', action='store_true', help='Do not launch the local Qt player after processing.')
    return p.parse_args()


def _build_bboxes_by_frame(raw_detections: Sequence[RawDetection]) -> dict[int, list[list[int]]]:
    bboxes: dict[int, list[list[int]]] = defaultdict(list)
    for d in raw_detections:
        bboxes[int(d.frame_idx)].append([int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2)])
    return dict(bboxes)


def _crop_detections_from_video(
    video_path: Path,
    raw_detections: Sequence[RawDetection],
    *,
    limit_frames: int | None,
) -> list[RawDetection]:
    dets_by_frame: dict[int, list[tuple[BoundingBox, float, Keypoint, Keypoint]]] = defaultdict(list)
    for d in raw_detections:
        dets_by_frame[int(d.frame_idx)].append((d.bbox, float(d.confidence), d.boom, d.mast_tip))

    out: list[RawDetection] = []
    with VideoReader(video_path) as reader:
        for frame_idx, frame in reader.read_frames():
            frame_idx = int(frame_idx)
            if limit_frames is not None and frame_idx >= int(limit_frames):
                break
            for bbox, confidence, boom, mast_tip in dets_by_frame.get(frame_idx, []):
                x1 = max(0, min(frame.shape[1], int(bbox.x1)))
                y1 = max(0, min(frame.shape[0], int(bbox.y1)))
                x2 = max(0, min(frame.shape[1], int(bbox.x2)))
                y2 = max(0, min(frame.shape[0], int(bbox.y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append(
                    RawDetection(
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=float(confidence),
                        frame_idx=frame_idx,
                        crop=frame[y1:y2, x1:x2],
                        boom=boom,
                        mast_tip=mast_tip,
                    )
                )
    return out


@cache_to_file('tmp/transforms')
def _compute_transforms(
    video_path: Path,
    *,
    stabilizer: StabilizerName,
    bboxes_by_frame: dict[int, list[list[int]]] | None,
    mask_margin_px: int,
    processing_max_dim: float | None,
    gmc_downscale: int,
    limit_frames: int | None,
    masked_vidstab_debug_dir: str | os.PathLike | None = None,
    masked_vidstab_debug_every_n: int = 1,
    masked_max_corners: int = int(STABLE_GFTT_MAX_CORNERS),
    masked_quality_level: float = float(STABLE_GFTT_QUALITY_LEVEL),
    masked_min_distance: float = float(STABLE_GFTT_MIN_DISTANCE),
    masked_block_size: int = int(STABLE_GFTT_BLOCK_SIZE),
) -> list[Transform]:
    if stabilizer == 'none':
        frames = get_video_properties(video_path).total_frames
        return [Transform(0, 0, 0, i) for i in range(frames + 1)]

    if stabilizer == 'vidstab':
        # VidStab reads the full video file; we don't currently truncate to `limit_frames`.
        if limit_frames is not None:
            print('WARNING: `--limit-frames` is ignored for stabilizer=vidstab (full video processed).')
        return compute_stabilization_transforms(video_path)

    if stabilizer == 'gmc':
        # Prefer the single-pass implementation that supports bbox masks.
        if limit_frames is not None:
            # compute_stabilization_transforms_gmc doesn't support limit_frames; do manual per-frame loop.
            gmc = GMC(downscale=int(gmc_downscale))
            out: list[Transform] = []
            with VideoReader(video_path) as reader:
                for frame_idx, frame in reader.read_frames():
                    frame_idx = int(frame_idx)
                    if frame_idx >= int(limit_frames):
                        break
                    excluded_bboxes = None if bboxes_by_frame is None else bboxes_by_frame.get(frame_idx, ())
                    out_t = gmc_transform_from_frame(
                        gmc, frame_idx=frame_idx, frame=frame, excluded_bboxes=excluded_bboxes
                    )
                    if out_t is not None:
                        out.append(out_t)
            return out

        return compute_stabilization_transforms_gmc(
            video_path,
            downscale=int(gmc_downscale),
            bboxes_by_frame=bboxes_by_frame,
        )

    if stabilizer == 'masked_vidstab':
        pmd = None if processing_max_dim is None else int(float(processing_max_dim))
        return compute_stabilization_transforms_masked_vidstab(
            video_path,
            bboxes_by_frame=bboxes_by_frame,
            mask_margin_px=int(mask_margin_px),
            processing_max_dim=pmd,
            max_corners=int(masked_max_corners),
            quality_level=float(masked_quality_level),
            min_distance=float(masked_min_distance),
            block_size=int(masked_block_size),
            limit_frames=limit_frames,
        )

    raise ValueError(f'Unknown stabilizer: {stabilizer}')


def _save_tracks_metadata(tracks: list[RenderableTrack], video_path: Path, output_dir: Path) -> Path:
    props = get_video_properties(video_path)

    metadata = Metadata(
        input_video_path=video_path.absolute().as_posix(),
        video_properties=VideoProperties(
            fps=float(props.fps),
            width=int(props.width),
            height=int(props.height),
            total_frames=int(props.total_frames),
        ),
        tracks=[
            TrackLite(
                track_id=int(track.track_id),
                start_frame=int(track.start_frame),
                end_frame=int(track.end_frame),
                start_time=float(track.start_frame / props.fps),
                duration=float(track.duration_frames / props.fps),
                detection_count=int(len(track.sorted_detections)),
                detections=[
                    DetectionLite(
                        frame_idx=int(det.frame_idx),
                        bbox=[int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)],
                        confidence=float(det.confidence),
                        interpolated=bool(det.interpolated),
                        boom=[float(det.boom.point.x), float(det.boom.point.y), float(det.boom.conf)],
                        mast_tip=[float(det.mast_tip.point.x), float(det.mast_tip.point.y), float(det.mast_tip.conf)],
                        anchor=[int(det.anchor.x), int(det.anchor.y)],
                        scale=float(det.scale),
                    )
                    for det in track.sorted_detections
                ],
            )
            for track in tracks
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{video_path.stem}.tracks.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(metadata, f)
    return out_path


def _save_stabilization_transforms(
    transforms_by_frame: list[dict[str, float | int]],
    *,
    frame_count: int,
    output_dir: Path,
    stem: str,
    stabilizer: StabilizerName,
    smoothing_window: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{stem}.stabilization_transforms.json'
    payload = {
        'frame_count': int(frame_count),
        'stabilizer': str(stabilizer),
        'smoothing_window': int(smoothing_window),
        'transforms': transforms_by_frame,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return out_path


def _save_raw_motion_transforms(
    transforms_by_frame: list[dict[str, float | int]],
    *,
    frame_count: int,
    output_dir: Path,
    stem: str,
    stabilizer: StabilizerName,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{stem}.raw_motion_transforms.json'
    payload = {
        'frame_count': int(frame_count),
        'stabilizer': str(stabilizer),
        'transforms': transforms_by_frame,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return out_path


def _dense_raw_motion_deltas_by_frame(
    *, raw_motion_transforms: list[Transform], frame_count: int
) -> list[dict[str, float | int]]:
    raw_by_frame: dict[int, Transform] = {int(t.frame_idx): t for t in raw_motion_transforms}
    out: list[dict[str, float | int]] = [{'frame_idx': 0, 'dx': 0.0, 'dy': 0.0, 'da': 0.0}]
    for frame_idx in range(1, int(frame_count)):
        t = raw_by_frame.get(int(frame_idx))
        if t is None:
            out.append({'frame_idx': int(frame_idx), 'dx': 0.0, 'dy': 0.0, 'da': 0.0})
        else:
            out.append({'frame_idx': int(frame_idx), 'dx': float(t.dx), 'dy': float(t.dy), 'da': float(t.da)})
    return out


def _bfill_rolling_mean(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] == 0:
        return arr
    n = max(1, min(int(n), int(arr.shape[0])))
    if n == 1:
        return arr
    pre_buffer = np.zeros(arr.shape[1], dtype=arr.dtype).reshape(1, arr.shape[1])
    post_buffer = np.zeros(arr.shape[1] * n, dtype=arr.dtype).reshape(n, arr.shape[1])
    arr_cumsum = np.cumsum(np.vstack((pre_buffer, arr, post_buffer)), axis=0)
    buffer_roll_mean = (arr_cumsum[n:, :] - arr_cumsum[:-n, :]) / float(n)
    trunc_roll_mean = buffer_roll_mean[:-n, :]
    bfill_size = arr.shape[0] - trunc_roll_mean.shape[0]
    bfill = np.tile(trunc_roll_mean[0, :], (bfill_size, 1))
    return np.vstack((bfill, trunc_roll_mean))


def _compute_stabilization_correction_by_frame(
    *,
    raw_motion_transforms: list[Transform],
    frame_count: int,
    smoothing_window: int,
) -> list[dict[str, float | int]]:
    # The Qt/web players expect per-frame *absolute* stabilization offsets (dx,dy,da)
    # applied directly when rendering frame k:
    #   correction[k] = smoothed_trajectory[k] - trajectory[k]
    # where trajectory is the cumulative sum of raw prev->curr motion deltas.
    if frame_count <= 0:
        return []

    raw_by_frame: dict[int, Transform] = {int(t.frame_idx): t for t in raw_motion_transforms}
    raw_deltas: list[list[float]] = []
    for frame_idx in range(1, frame_count):
        t = raw_by_frame.get(frame_idx)
        if t is None:
            raw_deltas.append([0.0, 0.0, 0.0])
        else:
            raw_deltas.append([float(t.dx), float(t.dy), float(t.da)])

    if not raw_deltas:
        return [{'frame_idx': i, 'dx': 0.0, 'dy': 0.0, 'da': 0.0} for i in range(frame_count)]

    raw = np.asarray(raw_deltas, dtype=np.float64)  # shape (frame_count-1,3)
    traj = np.cumsum(raw, axis=0)
    n = max(1, min(int(smoothing_window), int(traj.shape[0])))
    traj_s = _bfill_rolling_mean(traj, n=n)
    corr = traj_s - traj  # shape (frame_count-1,3)

    out: list[dict[str, float | int]] = [{'frame_idx': 0, 'dx': 0.0, 'dy': 0.0, 'da': 0.0}]
    for frame_idx in range(1, frame_count):
        dx, dy, da = corr[frame_idx - 1]
        out.append({'frame_idx': int(frame_idx), 'dx': float(dx), 'dy': float(dy), 'da': float(da)})
    return out


def run_local_pipeline(
    input_video: Path,
    *,
    output_dir: Path,
    skip_orientation: bool,
    orientation_model_path: str | None,
    yolo_model_path: str | None,
    stabilizer: StabilizerName,
    processing_max_dim: float | None,
    mask_margin_px: int,
    gmc_downscale: int,
    limit_frames: int | None,
    use_detector_crops: bool,
    masked_vidstab_debug_dir: str | os.PathLike | None,
    masked_vidstab_debug_every_n: int,
    smoothing_window: int,
    masked_max_corners: int,
    masked_quality_level: float,
    masked_min_distance: float,
    masked_block_size: int,
) -> LocalRunResult:
    output_dir = output_dir / input_video.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Orientation
    if skip_orientation:
        dominant_orientation = 0
        upright_video = input_video
    else:
        default_model = Path(__file__).resolve().parents[1] / 'inference' / 'weights' / 'orientation_fixer' / 'best.pt'
        model_path = str(default_model) if orientation_model_path is None else str(orientation_model_path)
        with timeit('orientation: detect'):
            fixer = OrientationFixer(model_path)
            dominant_orientation = int(fixer.detect_orientation(str(input_video)))
        upright_video = output_dir / f'{input_video.stem}.upright.mp4'
        with timeit('orientation: apply'):
            if dominant_orientation != 0:
                fixer.apply_rotation(str(input_video), str(upright_video), dominant_orientation)
            else:
                # Keep IO cheap: try hardlink; fall back to copy.
                if upright_video.exists():
                    upright_video.unlink()
                try:
                    os.link(input_video, upright_video)
                except OSError:
                    shutil.copy2(input_video, upright_video)

    # 2) Detection (YOLO)
    model = str(YOLO_MODEL_PATH) if yolo_model_path is None else str(yolo_model_path)
    with timeit('detection: yolo'):
        raw_detections = ObjectDetector(model).run_detection_pass(upright_video.as_posix())

    if limit_frames is not None:
        raw_detections = [d for d in raw_detections if int(d.frame_idx) < int(limit_frames)]

    bboxes_by_frame = _build_bboxes_by_frame(raw_detections)

    # 3) Transforms (stabilization / camera motion)
    with timeit(f'stabilization: {stabilizer}'):
        raw_motion_transforms = _compute_transforms(
            upright_video,
            stabilizer=stabilizer,
            bboxes_by_frame=bboxes_by_frame if bboxes_by_frame else None,
            mask_margin_px=int(mask_margin_px),
            processing_max_dim=processing_max_dim,
            gmc_downscale=int(gmc_downscale),
            limit_frames=limit_frames,
            masked_vidstab_debug_dir=masked_vidstab_debug_dir,
            masked_vidstab_debug_every_n=int(masked_vidstab_debug_every_n),
            masked_max_corners=int(masked_max_corners),
            masked_quality_level=float(masked_quality_level),
            masked_min_distance=float(masked_min_distance),
            masked_block_size=int(masked_block_size),
        )

    props = get_video_properties(upright_video)
    frame_count = int(props.total_frames)
    smoothing_window = 1 if frame_count <= 1 else max(1, min(int(smoothing_window), frame_count - 1))
    transforms_by_frame = _compute_stabilization_correction_by_frame(
        raw_motion_transforms=raw_motion_transforms,
        frame_count=frame_count,
        smoothing_window=smoothing_window,
    )

    # 4) Crops + embeddings (appearance)
    if use_detector_crops:
        raw_detections_with_crops = list(raw_detections)
    else:
        with timeit('appearance: crops'):
            # Re-crop on upright video to match the pipeline used in Modal tracking.
            raw_detections_with_crops = _crop_detections_from_video(
                upright_video, raw_detections, limit_frames=limit_frames
            )
    with timeit('appearance: embeddings'):
        detections: list[Detection] = EmbeddingExtractor(REID_MODEL_TYPE).run_embedding_pass(raw_detections_with_crops)

    # 5) Tracking + post-processing
    with timeit('tracking: trackers'):
        tracks: list[Track] = [Track(track_id=i, sorted_detections=[d]) for i, d in enumerate(detections)]
        trackers: Sequence = [
            TrackPreProcessor(),
            ILPTracker(video_path=upright_video.as_posix()),
            TrackPostProcessing(),
        ]
        for tracker in trackers:
            tracks = tracker.track(tracks, props, raw_motion_transforms)

    # 6) Write metadata for local Qt player
    with timeit('output: write metadata'):
        renderable_tracks = prepare_renderable_tracks(tracks, video_height=props.height)
        metadata_path = _save_tracks_metadata(renderable_tracks, upright_video, output_dir)
        stabilization_transforms_path = _save_stabilization_transforms(
            transforms_by_frame,
            frame_count=frame_count,
            output_dir=output_dir,
            stem=upright_video.stem,
            stabilizer=stabilizer,
            smoothing_window=smoothing_window,
        )
        raw_motion_transforms_path = _save_raw_motion_transforms(
            _dense_raw_motion_deltas_by_frame(raw_motion_transforms=raw_motion_transforms, frame_count=frame_count),
            frame_count=frame_count,
            output_dir=output_dir,
            stem=upright_video.stem,
            stabilizer=stabilizer,
        )

    print(f'Output directory: {output_dir}')
    print(f'Upright video: {upright_video}')
    print(f'Tracks metadata: {metadata_path}')
    print(f'Stabilization transforms: {stabilization_transforms_path}')
    print(f'Raw motion transforms: {raw_motion_transforms_path}')
    return LocalRunResult(
        output_dir=output_dir,
        upright_video_path=upright_video,
        metadata_path=metadata_path,
        stabilization_transforms_path=stabilization_transforms_path,
        raw_motion_transforms_path=raw_motion_transforms_path,
        dominant_orientation=dominant_orientation,
        transforms=raw_motion_transforms,
        tracks=tracks,
    )


def _launch_player(start_dir: Path) -> None:
    # Lazy import so the pipeline can run without PySide6.
    from inference.src.local_player_main import run_player

    run_player(start_dir=str(start_dir))


def main() -> int:
    args = _parse_args()
    result = run_local_pipeline(
        Path(args.input_video),
        output_dir=Path(args.output_dir),
        skip_orientation=bool(args.skip_orientation),
        orientation_model_path=args.orientation_model_path,
        yolo_model_path=args.yolo_model_path,
        stabilizer=str(args.stabilizer),  # type: ignore[arg-type]
        processing_max_dim=(float(args.processing_max_dim) if args.processing_max_dim is not None else None),
        mask_margin_px=int(args.mask_margin_px),
        gmc_downscale=int(args.gmc_downscale),
        limit_frames=int(args.limit_frames) if args.limit_frames is not None else None,
        use_detector_crops=bool(args.use_detector_crops),
        masked_vidstab_debug_dir=args.masked_vidstab_debug_dir,
        masked_vidstab_debug_every_n=int(args.masked_vidstab_debug_every_n),
        smoothing_window=int(args.smoothing_window),
        masked_max_corners=int(args.masked_max_corners),
        masked_quality_level=float(args.masked_quality_level),
        masked_min_distance=float(args.masked_min_distance),
        masked_block_size=int(args.masked_block_size),
    )

    if not args.no_player:
        _launch_player(result.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
