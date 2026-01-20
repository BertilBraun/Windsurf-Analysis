#!/usr/bin/env python3
"""
Compare VidStab kp methods (fixed black border).

Usage:
  python vidstab_kp_compare.py input.mp4
  python vidstab_kp_compare.py input.mp4 --windows 10 30 60
  python vidstab_kp_compare.py input.mp4 --kp GFTT BRISK DENSE FAST HARRIS MSER ORB STAR
  python vidstab_kp_compare.py input.mp4 --ext .avi --fourcc MJPG
  python vidstab_kp_compare.py input.mp4 --tracks-pkl run.tracks.pkl --mask-margin-px 20
  python vidstab_kp_compare.py input.mp4 --output-dir tmp --limit-frames 300
"""

from __future__ import annotations
import argparse, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
import json
import time

import cv2
import numpy as np

try:
    from vidstab import VidStab
except ImportError:
    from vidstab.VidStab import VidStab  # type: ignore


@dataclass(frozen=True)
class RunSpec:
    kp_method: str
    smoothing_window: int
    alpha: float
    processing_max_dim: float
    output_fourcc: str


@dataclass(frozen=True)
class TimingRecord:
    name: str
    seconds: float
    details: Dict[str, Any]


def sanitize_stem(s: str) -> str:
    return ''.join(c if (c.isalnum() or c in '-_') else '_' for c in s)


def sanitize_float_tag(x: float) -> str:
    s = f'{float(x):.3f}'.rstrip('0').rstrip('.')
    return s.replace('.', 'p').replace('-', 'm')


def _add_video_processing_to_path() -> None:
    video_processing_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(video_processing_dir))


def build_keypoint_mask(
    *,
    frame_shape: tuple[int, int, int] | tuple[int, int],
    excluded_bboxes: Sequence[Sequence[int]],
    margin_px: int = 0,
) -> np.ndarray:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    mask = np.full((height, width), 255, dtype=np.uint8)

    margin = max(0, int(margin_px))
    for bbox in excluded_bboxes:
        if len(bbox) != 4:
            raise ValueError(f'Invalid bbox: {bbox}')
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        x1 = max(0, min(width, x1 - margin))
        y1 = max(0, min(height, y1 - margin))
        x2 = max(0, min(width, x2 + margin))
        y2 = max(0, min(height, y2 + margin))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 0

    return mask


def load_bboxes_by_frame_from_tracks_pkl(tracks_pkl: Path) -> dict[int, list[list[int]]]:
    import pickle
    from collections import defaultdict

    _add_video_processing_to_path()
    from inference.src.player.core.player_state import Metadata  # noqa: E402

    with tracks_pkl.open('rb') as f:
        metadata = pickle.load(f)

    if not isinstance(metadata, Metadata):
        raise TypeError(f'Expected Metadata in {tracks_pkl}, got {type(metadata)}')

    by_frame: dict[int, list[list[int]]] = defaultdict(list)
    for track in metadata.tracks:
        for det in track.detections:
            if det.bbox is None:
                continue
            if len(det.bbox) != 4:
                continue
            by_frame[int(det.frame_idx)].append(
                [int(det.bbox[0]), int(det.bbox[1]), int(det.bbox[2]), int(det.bbox[3])]
            )

    return dict(by_frame)


def ensure_tracks_pkl(
    *,
    video_path: Path,
    output_dir: Path,
    tracks_pkl: Path | None,
    yolo_model_path: str | None,
    limit_frames: int | None,
) -> Path:
    _add_video_processing_to_path()
    from inference.src.player.core.player_state import DetectionLite, Metadata, TrackLite, VideoProperties  # noqa: E402
    from inference.src.tracking.detector import ObjectDetector  # noqa: E402
    from inference.src.util.video_io import get_video_properties  # noqa: E402
    from inference.src.settings import YOLO_MODEL_PATH  # noqa: E402

    if tracks_pkl is None:
        tracks_pkl = output_dir / f'{video_path.stem}.tracks.pkl'

    if tracks_pkl.exists():
        return tracks_pkl

    model_path = yolo_model_path or YOLO_MODEL_PATH
    detector = ObjectDetector(model_path)
    raw_detections = detector.run_detection_pass(video_path.as_posix())

    if limit_frames is not None:
        raw_detections = [d for d in raw_detections if int(d.frame_idx) < int(limit_frames)]

    props = get_video_properties(video_path)
    fps = float(props.fps) if float(props.fps) > 0 else 30.0

    tracks: list[TrackLite] = []
    for i, det in enumerate(raw_detections):
        frame_idx = int(det.frame_idx)
        x1, y1, x2, y2 = (int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2))
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)
        boom = [float(det.boom.point.x), float(det.boom.point.y), float(det.boom.conf)]
        mast_tip = [float(det.mast_tip.point.x), float(det.mast_tip.point.y), float(det.mast_tip.conf)]
        if float(det.boom.conf) >= 0.3:
            anchor = [int(det.boom.point.x), int(det.boom.point.y)]
        else:
            anchor = [int(cx), int(cy)]
        tracks.append(
            TrackLite(
                track_id=int(i),
                start_frame=frame_idx,
                end_frame=frame_idx,
                start_time=float(frame_idx / fps),
                duration=float(1.0 / fps),
                detection_count=1,
                detections=[
                    DetectionLite(
                        frame_idx=frame_idx,
                        bbox=[x1, y1, x2, y2],
                        confidence=float(det.confidence),
                        interpolated=False,
                        boom=boom,
                        mast_tip=mast_tip,
                        anchor=anchor,
                        scale=1.0,
                    )
                ],
            )
        )

    metadata = Metadata(
        input_video_path=video_path.absolute().as_posix(),
        video_properties=VideoProperties(
            fps=fps,
            width=int(props.width),
            height=int(props.height),
            total_frames=int(props.total_frames),
        ),
        tracks=tracks,
    )

    tracks_pkl.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with tracks_pkl.open('wb') as f:
        pickle.dump(metadata, f)

    return tracks_pkl


class BBoxMaskedVidStab(VidStab):
    def __init__(
        self,
        *,
        bboxes_by_frame: Mapping[int, Sequence[Sequence[int]]] | None,
        mask_margin_px: int = 0,
        min_kps_for_mask: int = 30,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._bboxes_by_frame = dict(bboxes_by_frame or {})
        self._mask_margin_px = int(mask_margin_px)
        self._min_kps_for_mask = max(0, int(min_kps_for_mask))
        self._alpha = float(alpha)

    def _gen_transforms(self):  # type: ignore[override]
        from vidstab import general_utils  # local import to avoid changing global vidstab imports

        self.trajectory = np.array(self._trajectory)
        self.smoothed_trajectory = general_utils.bfill_rolling_mean(self.trajectory, n=self._smoothing_window)

        raw = np.array(self._raw_transforms)
        correction = self.smoothed_trajectory - self.trajectory
        self.transforms = raw + (float(self._alpha) * correction)

        # Dump superfluous frames (same behavior as vidstab.VidStab)
        # noinspection PyProtectedMember
        n = self.frame_queue._max_frames
        if n:
            self.trajectory = self.trajectory[: n - 1, :]
            self.smoothed_trajectory = self.smoothed_trajectory[: n - 1, :]
            self.transforms = self.transforms[: n - 1, :]

    def _mask_proc_for_frame(
        self, *, frame_idx: int, frame_shape: tuple[int, int, int], gray_proc: np.ndarray
    ) -> np.ndarray | None:
        bboxes = self._bboxes_by_frame.get(int(frame_idx), ())
        if not bboxes:
            return None
        mask_full = build_keypoint_mask(frame_shape=frame_shape, excluded_bboxes=bboxes, margin_px=self._mask_margin_px)
        if (mask_full.shape[0], mask_full.shape[1]) != (gray_proc.shape[0], gray_proc.shape[1]):
            mask_full = cv2.resize(
                mask_full, (int(gray_proc.shape[1]), int(gray_proc.shape[0])), interpolation=cv2.INTER_NEAREST
            )
        if mask_full.dtype != np.uint8:
            mask_full = mask_full.astype(np.uint8, copy=False)
        return mask_full

    def _detect_kps(self, gray_proc: np.ndarray, mask_proc: np.ndarray | None) -> np.ndarray:
        if mask_proc is not None and mask_proc.dtype != np.uint8:
            mask_proc = mask_proc.astype(np.uint8, copy=False)
        try:
            prev_kps = self.kp_detector.detect(gray_proc, mask_proc)
        except TypeError:
            if mask_proc is not None:
                gray_proc = gray_proc.copy()
                gray_proc[mask_proc == 0] = 0
            prev_kps = self.kp_detector.detect(gray_proc)

        # noinspection PyArgumentList
        if prev_kps is None:
            return np.empty((0, 1, 2), dtype='float32')
        return np.array([kp.pt for kp in prev_kps], dtype='float32').reshape(-1, 1, 2)

    def _detect_kps_with_mask_fallback(self, gray_proc: np.ndarray, mask_proc: np.ndarray | None) -> np.ndarray:
        if mask_proc is None or self._min_kps_for_mask <= 0:
            return self._detect_kps(gray_proc, mask_proc)

        masked = self._detect_kps(gray_proc, mask_proc)
        if masked.shape[0] >= self._min_kps_for_mask:
            return masked

        return self._detect_kps(gray_proc, None)

    def _process_first_frame(self, array=None):  # type: ignore[override]
        _, _, _ = self.frame_queue.read_frame(array=array, pop_ind=False)
        if array is None and len(self.frame_queue.frames) == 0:
            raise ValueError('First frame is None. Check if input file/stream is correct.')

        prev_frame = self.frame_queue.frames[-1]
        prev_frame_gray = self._resize_frame(prev_frame.gray_image)
        frame_idx = int(self.frame_queue.inds[-1]) if len(self.frame_queue.inds) else 0
        mask_proc = self._mask_proc_for_frame(
            frame_idx=frame_idx, frame_shape=prev_frame.image.shape, gray_proc=prev_frame_gray
        )

        self.prev_kps = self._detect_kps_with_mask_fallback(prev_frame_gray, mask_proc)
        self.prev_gray = prev_frame_gray[:]

    def _update_prev_frame(self, current_frame_gray, *, frame_idx: int, frame_shape: tuple[int, int, int]):  # type: ignore[override]
        self.prev_gray = current_frame_gray[:]
        mask_proc = self._mask_proc_for_frame(
            frame_idx=frame_idx, frame_shape=frame_shape, gray_proc=current_frame_gray
        )
        self.prev_kps = self._detect_kps_with_mask_fallback(current_frame_gray, mask_proc)

    def _gen_next_raw_transform(self):  # type: ignore[override]
        current_frame = self.frame_queue.frames[-1]
        current_frame_gray = self._resize_frame(current_frame.gray_image)

        if self.prev_kps is None or self.prev_kps.size == 0:
            transform_i = [0.0, 0.0, 0.0]
            current_frame_idx = int(self.frame_queue.inds[-1]) if len(self.frame_queue.inds) else 0
            self._update_prev_frame(
                current_frame_gray, frame_idx=current_frame_idx, frame_shape=current_frame.image.shape
            )
            self._raw_transforms.append(transform_i[:])
            self._update_trajectory(transform_i)
            return

        optical_flow = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_frame_gray, self.prev_kps, None)

        from vidstab import vidstab_utils  # local import to avoid changing global vidstab imports

        matched_keypoints = vidstab_utils.match_keypoints(optical_flow, self.prev_kps)
        if self._min_kps_for_mask > 0 and len(matched_keypoints[0]) < self._min_kps_for_mask:
            transform_i = [0.0, 0.0, 0.0]
        else:
            transform_i = vidstab_utils.estimate_partial_transform(matched_keypoints)

        current_frame_idx = int(self.frame_queue.inds[-1]) if len(self.frame_queue.inds) else 0
        self._update_prev_frame(current_frame_gray, frame_idx=current_frame_idx, frame_shape=current_frame.image.shape)
        self._raw_transforms.append(transform_i[:])
        self._update_trajectory(transform_i)


def stabilize_once(
    input_path: Path,
    output_path: Path,
    spec: RunSpec,
    playback: bool,
    show_progress: bool,
    *,
    bboxes_by_frame: Mapping[int, Sequence[Sequence[int]]] | None,
    mask_margin_px: int,
    min_kps_for_mask: int,
    border_size: int | str,
) -> None:
    stab = BBoxMaskedVidStab(
        kp_method=spec.kp_method,
        alpha=float(spec.alpha),
        processing_max_dim=spec.processing_max_dim,
        bboxes_by_frame=bboxes_by_frame,
        mask_margin_px=int(mask_margin_px),
        min_kps_for_mask=int(min_kps_for_mask),
    )
    stab.stabilize(
        input_path=str(input_path),
        output_path=str(output_path),
        smoothing_window=spec.smoothing_window,
        border_type='black',
        border_size=border_size,
        layer_func=None,
        playback=playback,
        show_progress=show_progress,
        output_fourcc=spec.output_fourcc,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--ext', default='.mp4')
    ap.add_argument('--fourcc', default='mp4v')
    ap.add_argument('--kp', nargs='*', default=['GFTT', 'BRISK', 'FAST', 'MSER', 'ORB'])
    ap.add_argument('--windows', nargs='*', type=int, default=[30])
    ap.add_argument('--alpha', nargs='*', type=float, default=[1.0], help='Stabilization strength (0=no stabilize, 1=full).')
    ap.add_argument('--max-dim', type=float, default=float('inf'))
    ap.add_argument(
        '--border-size',
        type=str,
        default='auto',
        help="Border handling size: integer pixels or 'auto' (min padding to avoid cutting off transformed frames).",
    )
    ap.add_argument(
        '--tracks-pkl',
        type=str,
        default=None,
        help='Optional `.tracks.pkl` file; if missing, one is created in --output-dir from YOLO detections.',
    )
    ap.add_argument(
        '--yolo-model-path',
        type=str,
        default=None,
        help='Optional override for YOLO weights used when generating `.tracks.pkl`.',
    )
    ap.add_argument(
        '--limit-frames',
        type=int,
        default=None,
        help='Optionally limit detections used for masking and `.tracks.pkl` generation.',
    )
    ap.add_argument('--mask-margin-px', type=int, default=20, help='Padding around detected bboxes when masking.')
    ap.add_argument(
        '--min-kps-for-mask',
        type=int,
        default=30,
        help='If masked keypoints fall below this, fall back to unmasked keypoints for that frame.',
    )
    ap.add_argument(
        '--timings-json',
        type=str,
        default=None,
        help='Optional path to write per-stage timing log as JSON.',
    )
    ap.add_argument('--playback', action='store_true')
    ap.add_argument('--no-progress', action='store_true')
    args = ap.parse_args()

    input_path = Path(args.video).expanduser().resolve()
    if not input_path.exists():
        print(f'[ERROR] Not found: {input_path}', file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = args.ext if args.ext.startswith('.') else f'.{args.ext}'
    base = sanitize_stem(input_path.stem)

    border_size: int | str
    if str(args.border_size).lower() == 'auto':
        border_size = 'auto'
    else:
        try:
            border_size = int(args.border_size)
        except ValueError:
            print(f"[ERROR] --border-size must be int or 'auto' (got {args.border_size!r})", file=sys.stderr)
            return 2

    timings: list[TimingRecord] = []

    tracks_pkl_arg = Path(args.tracks_pkl).expanduser().resolve() if args.tracks_pkl else None
    t0 = time.perf_counter()
    tracks_pkl = ensure_tracks_pkl(
        video_path=input_path,
        output_dir=out_dir,
        tracks_pkl=tracks_pkl_arg,
        yolo_model_path=args.yolo_model_path,
        limit_frames=args.limit_frames,
    )
    timings.append(
        TimingRecord(
            name='ensure_tracks_pkl',
            seconds=float(time.perf_counter() - t0),
            details={
                'tracks_pkl': str(tracks_pkl),
                'limit_frames': args.limit_frames,
                'provided_tracks_pkl': str(tracks_pkl_arg) if tracks_pkl_arg else None,
                'yolo_model_path': args.yolo_model_path,
            },
        )
    )

    t0 = time.perf_counter()
    bboxes_by_frame = load_bboxes_by_frame_from_tracks_pkl(tracks_pkl)
    if args.limit_frames is not None:
        bboxes_by_frame = {int(k): v for k, v in bboxes_by_frame.items() if int(k) < int(args.limit_frames)}
    timings.append(
        TimingRecord(
            name='load_bboxes_by_frame',
            seconds=float(time.perf_counter() - t0),
            details={'frames_with_bboxes': int(len(bboxes_by_frame))},
        )
    )
    print(f'Loaded bboxes for {len(bboxes_by_frame)} frames from {tracks_pkl.name}')

    run_specs: List[RunSpec] = [
        RunSpec(
            kp_method=kp,
            smoothing_window=w,
            alpha=float(a),
            processing_max_dim=args.max_dim,
            output_fourcc=args.fourcc,
        )
        for kp in args.kp
        for w in args.windows
        for a in args.alpha
    ]

    ok, fail = [], []
    mask_tag = f'__mask-dets-m{int(args.mask_margin_px)}'
    border_tag = f'__border-{("auto" if border_size == "auto" else f"px{int(border_size)}")}'
    total_t0 = time.perf_counter()
    for i, spec in enumerate(run_specs, start=1):
        alpha_tag = f'__a-{sanitize_float_tag(spec.alpha)}'
        out_name = (
            f'{base}__kp-{spec.kp_method}__win-{spec.smoothing_window}{alpha_tag}{mask_tag}{border_tag}__black{ext}'
        )
        out_path = out_dir / out_name
        print(f'[{i:02d}/{len(run_specs):02d}] {out_name}')
        t0 = time.perf_counter()
        try:
            stabilize_once(
                input_path,
                out_path,
                spec,
                args.playback,
                not args.no_progress,
                bboxes_by_frame=bboxes_by_frame,
                mask_margin_px=int(args.mask_margin_px),
                min_kps_for_mask=int(args.min_kps_for_mask),
                border_size=border_size,
            )
            elapsed = float(time.perf_counter() - t0)
            timings.append(
                TimingRecord(
                    name='stabilize',
                    seconds=elapsed,
                    details={
                        'output': str(out_path),
                        'kp_method': spec.kp_method,
                        'smoothing_window': int(spec.smoothing_window),
                        'alpha': float(spec.alpha),
                        'processing_max_dim': float(spec.processing_max_dim),
                        'mask_margin_px': int(args.mask_margin_px),
                        'min_kps_for_mask': int(args.min_kps_for_mask),
                        'border_size': border_size,
                        'border_type': 'black',
                        'fourcc': spec.output_fourcc,
                    },
                )
            )
            if out_path.exists() and out_path.stat().st_size > 0:
                ok.append(out_path)
                print(f'  -> OK ({elapsed:.2f}s)\n')
            else:
                fail.append((spec, 'no output written'))
                print(f'  -> FAIL (no output) ({elapsed:.2f}s)\n')
        except Exception as e:
            fail.append((spec, repr(e)))
            elapsed = float(time.perf_counter() - t0)
            timings.append(
                TimingRecord(
                    name='stabilize_failed',
                    seconds=elapsed,
                    details={
                        'kp_method': spec.kp_method,
                        'smoothing_window': int(spec.smoothing_window),
                        'alpha': float(spec.alpha),
                        'error': repr(e),
                    },
                )
            )
            print(f'  -> FAIL {e!r} ({elapsed:.2f}s)\n')

    timings.append(
        TimingRecord(name='total', seconds=float(time.perf_counter() - total_t0), details={'runs': int(len(run_specs))})
    )

    print('=== Summary ===')
    print(f'Success: {len(ok)}')
    for p in ok:
        print(f'  - {p.name}')
    print(f'\nFailed: {len(fail)}')
    for spec, err in fail:
        print(f'  - kp={spec.kp_method} win={spec.smoothing_window}: {err}')
    if fail:
        print('\nCodec tip: if mp4 fails, try: --ext .avi --fourcc MJPG')

    if args.timings_json:
        out = Path(args.timings_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8') as f:
            json.dump(
                {
                    'video': str(input_path),
                    'output_dir': str(out_dir),
                    'timings': [{'name': t.name, 'seconds': t.seconds, 'details': t.details} for t in timings],
                },
                f,
                indent=2,
            )
        print(f'\nWrote timings: {out}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
