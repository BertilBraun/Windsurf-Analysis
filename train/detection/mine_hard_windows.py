#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

try:
    from hard_windows import HardWindow, save_hard_windows
except ImportError:
    from train.detection.hard_windows import HardWindow, save_hard_windows


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_processing.scripts import local_modal_pipeline_player as local_pipeline
from video_processing.inference.src.common_types import Keypoint, RenderableDetection, RenderableTrack


VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv'}
KP_VISIBLE_CONF = 0.15


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Run the full local modal pipeline per video, then mine windows where boom/mast-tip keypoints jitter '
            'strongly relative to the smoothed render anchor over short horizons.'
        )
    )
    p.add_argument('--videos', type=Path, required=True, help='Video file or directory (recursively scanned).')
    p.add_argument('--out', type=Path, required=True, help='Output .txt file listing hard windows.')
    p.add_argument(
        '--window-size',
        type=int,
        default=20,
        help='Window size in frames around the peak jitter frame.',
    )
    p.add_argument(
        '--top-windows',
        type=int,
        default=200,
        help='Max number of non-overlapping windows to keep.',
    )
    p.add_argument(
        '--min-window-score',
        type=float,
        default=8.0,
        help='Discard windows whose average jitter score is below this threshold (pixels).',
    )
    p.add_argument(
        '--relative-paths',
        action='store_true',
        help='Write video paths relative to the output file when possible.',
    )
    return p.parse_args()


def _collect_videos(videos_root: Path) -> list[Path]:
    p = Path(videos_root)
    if p.is_file():
        return [p] if p.suffix.lower() in VIDEO_EXTS else []
    return [x for x in sorted(p.rglob('*')) if x.is_file() and x.suffix.lower() in VIDEO_EXTS]


def _keypoint_visible(kp: Keypoint) -> bool:
    return float(kp.conf) >= KP_VISIBLE_CONF


def _offset_from_anchor(det: RenderableDetection, kp: Keypoint) -> tuple[float, float]:
    return (float(kp.point.x) - float(det.anchor.x), float(kp.point.y) - float(det.anchor.y))


def _predict_offset(prev2: tuple[float, float], prev1: tuple[float, float]) -> tuple[float, float]:
    return (
        float(prev1[0]) + (float(prev1[0]) - float(prev2[0])),
        float(prev1[1]) + (float(prev1[1]) - float(prev2[1])),
    )


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _score_track_frame_jitter(track: RenderableTrack) -> dict[int, tuple[float, str]]:
    scores: dict[int, tuple[float, str]] = {}
    dets = track.sorted_detections
    if len(dets) < 3:
        return scores

    for i in range(2, len(dets)):
        d0 = dets[i - 2]
        d1 = dets[i - 1]
        d2 = dets[i]

        parts: list[tuple[str, float]] = []
        for name, kp0, kp1, kp2 in (
            ('boom', d0.boom, d1.boom, d2.boom),
            ('tip', d0.mast_tip, d1.mast_tip, d2.mast_tip),
        ):
            if not (_keypoint_visible(kp0) and _keypoint_visible(kp1) and _keypoint_visible(kp2)):
                continue
            off0 = _offset_from_anchor(d0, kp0)
            off1 = _offset_from_anchor(d1, kp1)
            off2 = _offset_from_anchor(d2, kp2)
            pred = _predict_offset(off0, off1)
            residual = _distance(off2, pred)
            parts.append((name, float(residual)))

        if not parts:
            continue

        score = max(val for _name, val in parts)
        notes = ','.join([f'track={int(track.track_id)}'] + [f'{name}={val:.1f}' for name, val in parts])
        frame_idx = int(d2.frame_idx)
        prev = scores.get(frame_idx)
        if prev is None or float(score) > float(prev[0]):
            scores[frame_idx] = (float(score), notes)

    return scores


def _score_renderable_tracks(renderable_tracks: list[RenderableTrack]) -> list[dict]:
    merged: dict[int, tuple[float, str]] = {}
    for track in tqdm(renderable_tracks, desc='Scoring tracks', leave=False):
        per_track = _score_track_frame_jitter(track)
        for frame_idx, payload in per_track.items():
            prev = merged.get(int(frame_idx))
            if prev is None or float(payload[0]) > float(prev[0]):
                merged[int(frame_idx)] = payload

    return [
        {'frame_idx': int(frame_idx), 'score': float(score), 'notes': str(notes)}
        for frame_idx, (score, notes) in sorted(merged.items())
    ]


def _build_windows(
    video_path: Path,
    scored_frames: list[dict],
    *,
    window_size: int,
    min_window_score: float,
) -> list[HardWindow]:
    if not scored_frames:
        return []

    half_window = max(1, int(window_size) // 2)
    windows: list[HardWindow] = []
    for item in scored_frames:
        peak_frame = int(item['frame_idx'])
        lo = peak_frame - half_window
        hi = peak_frame + half_window
        neighborhood = [x for x in scored_frames if lo <= int(x['frame_idx']) <= hi]
        if not neighborhood:
            continue
        score = sum(float(x['score']) for x in neighborhood) / float(len(neighborhood))
        if score < float(min_window_score):
            continue
        start_frame = max(0, lo)
        end_frame = max(start_frame, hi)
        windows.append(
            HardWindow(
                video_path=video_path.resolve(),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                peak_frame=int(peak_frame),
                score=float(score),
                notes=str(item.get('notes', ''))[:240],
            )
        )
    return windows


def _select_top_non_overlapping(windows: list[HardWindow], limit: int) -> list[HardWindow]:
    selected: list[HardWindow] = []
    by_video: dict[str, list[HardWindow]] = {}

    for window in sorted(windows, key=lambda w: float(w.score), reverse=True):
        video_key = str(window.video_path.resolve())
        overlaps = False
        for existing in by_video.get(video_key, []):
            if not (int(window.end_frame) < int(existing.start_frame) or int(window.start_frame) > int(existing.end_frame)):
                overlaps = True
                break
        if overlaps:
            continue
        by_video.setdefault(video_key, []).append(window)
        selected.append(window)
        if len(selected) >= max(1, int(limit)):
            break

    return sorted(selected, key=lambda w: (str(w.video_path), int(w.start_frame)))


def _run_pipeline_for_video(video_path: Path, pipeline_root: Path):
    return local_pipeline.run_local_pipeline(
        video_path,
        output_dir=pipeline_root,
        skip_orientation=False,
        overwrite_input_with_upright=True,
        write_outputs=False,
        orientation_model_path=None,
        yolo_model_path=None,
        stabilizer='masked_vidstab',
        processing_max_dim=None,
        mask_margin_px=20,
        gmc_downscale=2,
        limit_frames=None,
        use_detector_crops=False,
        masked_vidstab_debug_dir=None,
        masked_vidstab_debug_every_n=1,
        smoothing_window=int(local_pipeline.STABLE_SMOOTHING_WINDOW),
        masked_max_corners=int(local_pipeline.STABLE_GFTT_MAX_CORNERS),
        masked_quality_level=float(local_pipeline.STABLE_GFTT_QUALITY_LEVEL),
        masked_min_distance=float(local_pipeline.STABLE_GFTT_MIN_DISTANCE),
        masked_block_size=int(local_pipeline.STABLE_GFTT_BLOCK_SIZE),
        render_keypoints=False,
        render_keypoints_source='raw',
        render_keypoints_output=None,
        render_keypoints_every_n=1,
        render_keypoints_min_det_conf=0.0,
    )


def main() -> int:
    args = _parse_args()

    videos_root = Path(args.videos)
    out_path = Path(args.out)
    if not videos_root.exists():
        raise SystemExit(f'--videos does not exist: {videos_root}')

    videos = _collect_videos(videos_root)
    if not videos:
        raise SystemExit(f'No videos found under: {videos_root}')

    all_windows: list[HardWindow] = []
    videos_scored = 0
    frames_with_scores = 0
    pipeline_root = (out_path.parent if out_path.parent != Path('') else Path('.')) / '.hard_windows_pipeline'
    pipeline_root.mkdir(parents=True, exist_ok=True)
    for video_path in tqdm(videos, desc='Mining videos'):
        result = _run_pipeline_for_video(video_path, pipeline_root)
        scored_frames = _score_renderable_tracks(result.renderable_tracks)
        video_windows = _build_windows(
            video_path,
            scored_frames,
            window_size=int(args.window_size),
            min_window_score=float(args.min_window_score),
        )
        frames_with_scores += len(scored_frames)
        all_windows.extend(
            video_windows
        )
        selected_so_far = _select_top_non_overlapping(all_windows, int(args.top_windows))
        save_hard_windows(out_path, selected_so_far, relative_to_file=bool(args.relative_paths))
        print(
            f'Processed {video_path.name}: '
            f'frames_with_scores={len(scored_frames)} windows={len(video_windows)} '
            f'top_written={len(selected_so_far)}'
        )
        del selected_so_far
        del video_windows
        videos_scored += 1
        del scored_frames
        del result
        gc.collect()

    selected = _select_top_non_overlapping(all_windows, int(args.top_windows))
    save_hard_windows(out_path, selected, relative_to_file=bool(args.relative_paths))

    print(
        yaml.safe_dump(
            {
                'videos_found': len(videos),
                'videos_scored': videos_scored,
                'frames_with_scores': frames_with_scores,
                'windows_candidates': len(all_windows),
                'windows_written': len(selected),
                'output_file': str(out_path.resolve()),
                'score_mode': 'anchor_relative_short_horizon_jitter',
            },
            sort_keys=False,
        ).strip()
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
