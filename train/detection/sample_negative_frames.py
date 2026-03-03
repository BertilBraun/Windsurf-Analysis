#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


SUPPORTED_VIDEO_EXTS = {'.mp4', '.mov', '.m4v', '.avi', '.mkv'}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Randomly sample full-frame negative images from videos and write empty YOLO labels.'
    )
    p.add_argument('videos', type=Path, help='Directory containing negative videos (scanned recursively).')
    p.add_argument(
        '--out',
        type=Path,
        default=Path('train/detection/windsurf_dataset'),
        help='Output dataset directory for .jpg and empty .txt files.',
    )
    p.add_argument('--max-samples', type=int, default=50, help='Maximum number of negative frames to write.')
    p.add_argument('--seed', type=int, default=0, help='RNG seed.')
    return p.parse_args()


def _collect_videos(root: Path) -> list[Path]:
    videos: list[Path] = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTS:
            videos.append(path)
    return videos


def _read_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(0, total)


def _safe_stem(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in ('-', '_'):
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out).strip('_') or 'negative'


def _next_output_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    img_path = out_dir / f'{stem}.jpg'
    txt_path = out_dir / f'{stem}.txt'
    if not img_path.exists() and not txt_path.exists():
        return img_path, txt_path

    idx = 1
    while True:
        img_path = out_dir / f'{stem}__dup{idx:02d}.jpg'
        txt_path = out_dir / f'{stem}__dup{idx:02d}.txt'
        if not img_path.exists() and not txt_path.exists():
            return img_path, txt_path
        idx += 1


def main() -> None:
    args = _parse_args()
    videos_root = Path(args.videos)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = _collect_videos(videos_root)
    if not videos:
        raise SystemExit(f'No videos found under: {videos_root}')

    weighted_videos: list[tuple[Path, int]] = []
    total_frames = 0
    for video_path in videos:
        frame_count = _read_frame_count(video_path)
        if frame_count <= 0:
            continue
        weighted_videos.append((video_path, frame_count))
        total_frames += frame_count

    if not weighted_videos or total_frames <= 0:
        raise SystemExit(f'No readable video frames found under: {videos_root}')

    rng = random.Random(int(args.seed))
    target = max(0, int(args.max_samples))
    seen: set[tuple[Path, int]] = set()
    picks: list[tuple[Path, int]] = []
    max_attempts = max(target * 20, 100)

    for _ in range(max_attempts):
        if len(picks) >= target:
            break
        video_path, frame_count = rng.choices(
            [item[0] for item in weighted_videos],
            weights=[item[1] for item in weighted_videos],
            k=1,
        )[0], 0
        for candidate_path, candidate_count in weighted_videos:
            if candidate_path == video_path:
                frame_count = candidate_count
                break
        if frame_count <= 0:
            continue
        frame_idx = rng.randint(0, frame_count - 1)
        key = (video_path, frame_idx)
        if key in seen:
            continue
        seen.add(key)
        picks.append(key)

    if not picks:
        raise SystemExit('Failed to select any unique frames.')

    picks_by_video: dict[Path, list[int]] = {}
    for video_path, frame_idx in picks:
        picks_by_video.setdefault(video_path, []).append(frame_idx)

    written = 0
    for video_path, frame_indices in picks_by_video.items():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        for frame_idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            stem = _safe_stem(f'neg_{video_path.stem}_frame_{frame_idx:06d}')
            img_path, txt_path = _next_output_paths(out_dir, stem)
            if not cv2.imwrite(str(img_path), frame):
                continue
            txt_path.write_text('', encoding='utf-8')
            written += 1

        cap.release()

    print(f'videos_found: {len(videos)}')
    print(f'frames_requested: {target}')
    print(f'frames_selected: {len(picks)}')
    print(f'frames_written: {written}')
    print(f'output_dir: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
