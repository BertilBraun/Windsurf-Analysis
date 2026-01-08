#!/usr/bin/env python3
"""
Grid-search VidStab(GFTT) hyperparameters and run a blind tournament comparison.

Usage:
  python video_processing/scripts/vidstab_gftt_tournament.py input.mp4 --mode all
  python video_processing/scripts/vidstab_gftt_tournament.py input.mp4 --mode precompute
  python video_processing/scripts/vidstab_gftt_tournament.py input.mp4 --mode tournament

Keys during tournament:
  - 1: pick left
  - 2: pick right
  - r: restart current matchup
  - space: pause/resume
  - q or ESC: save + quit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _import_vidstab_comparison():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import vidstab_comparison as vc  # type: ignore

    return vc


vc = _import_vidstab_comparison()


@dataclass(frozen=True)
class GFTTSpec:
    processing_max_dim: float
    smoothing_window: int
    max_corners: int
    quality_level: float
    min_distance: float
    block_size: int
    use_masking: bool
    output_fourcc: str


def _float_tag(x: float) -> str:
    if np.isinf(x):
        return 'inf'
    s = f'{float(x):g}'
    return s.replace('.', 'p').replace('-', 'm')


def _spec_id(spec: GFTTSpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha1(payload).hexdigest()[:12]


def _get_video_wh(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Failed to open: {video_path}')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f'Failed to read first frame: {video_path}')
            height, width = int(frame.shape[0]), int(frame.shape[1])
        return width, height
    finally:
        cap.release()


def stabilize_once(
    *,
    input_path: Path,
    output_path: Path,
    spec: GFTTSpec,
    show_progress: bool,
    playback: bool,
    bboxes_by_frame: Mapping[int, Sequence[Sequence[int]]] | None,
    mask_margin_px: int,
    min_kps_for_mask: int,
) -> None:
    stab = vc.BBoxMaskedVidStab(
        kp_method='GFTT',
        processing_max_dim=spec.processing_max_dim,
        bboxes_by_frame=bboxes_by_frame if spec.use_masking else None,
        mask_margin_px=int(mask_margin_px),
        min_kps_for_mask=int(min_kps_for_mask),
        maxCorners=int(spec.max_corners),
        qualityLevel=float(spec.quality_level),
        minDistance=float(spec.min_distance),
        blockSize=int(spec.block_size),
    )
    stab.stabilize(
        input_path=str(input_path),
        output_path=str(output_path),
        smoothing_window=int(spec.smoothing_window),
        border_type='black',
        border_size=0,
        layer_func=None,
        playback=bool(playback),
        show_progress=bool(show_progress),
        output_fourcc=str(spec.output_fourcc),
    )


def _default_run_dir(input_path: Path) -> Path:
    return input_path.parent / f'{vc.sanitize_stem(input_path.stem)}__gftt_tournament'


def _safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _next_power_of_two_below(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _deterministic_lr(seed: int, match_key: str, a: str, b: str) -> tuple[str, str]:
    h = hashlib.sha1(f'{seed}:{match_key}:{a}:{b}'.encode('utf-8')).digest()
    return (a, b) if (h[0] % 2 == 0) else (b, a)


def _read_frame(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = int(img.shape[0]), int(img.shape[1])
    if h == target_h:
        return img
    scale = float(target_h) / float(h)
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def _resize_by_scale(img: np.ndarray, scale: float) -> np.ndarray:
    s = float(scale)
    if s == 1.0:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def _stack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    target_h = min(int(left.shape[0]), int(right.shape[0]))
    left_r = _resize_to_height(left, target_h)
    right_r = _resize_to_height(right, target_h)
    return np.concatenate([left_r, right_r], axis=1)


def _play_match(*, left_video: Path, right_video: Path, title: str, viewer_scale: float) -> int | None:
    cap_l = cv2.VideoCapture(str(left_video))
    cap_r = cv2.VideoCapture(str(right_video))
    try:
        if not cap_l.isOpened() or not cap_r.isOpened():
            raise RuntimeError('Failed to open one of the match videos.')

        fps_l = float(cap_l.get(cv2.CAP_PROP_FPS))
        fps_r = float(cap_r.get(cv2.CAP_PROP_FPS))
        fps = min(fps_l if fps_l > 0 else 30.0, fps_r if fps_r > 0 else 30.0)
        delay_ms = max(1, int(round(1000.0 / fps)))

        paused = False
        while True:
            if not paused:
                frame_l = _read_frame(cap_l)
                frame_r = _read_frame(cap_r)
            else:
                frame_l = frame_r = None

            if frame_l is None or frame_r is None:
                cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_l = _read_frame(cap_l)
                frame_r = _read_frame(cap_r)
                if frame_l is None or frame_r is None:
                    return None

            view = _stack_side_by_side(frame_l, frame_r)
            view = _resize_by_scale(view, float(viewer_scale))
            overlay = view.copy()
            cv2.putText(overlay, '1:LEFT', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, '2:RIGHT', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(
                overlay,
                'r:RESTART  space:PAUSE  q/ESC:QUIT',
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(overlay, title, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.8, view, 0.2, 0.0, view)
            cv2.imshow('GFTT Tournament', view)

            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key in (27, ord('q')):  # ESC / q
                return None
            if key == ord('1'):
                return 0
            if key == ord('2'):
                return 1
            if key == ord('r'):
                cap_l.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap_r.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
                continue
            if key == ord(' '):
                paused = not paused
                continue
    finally:
        cap_l.release()
        cap_r.release()
        try:
            cv2.destroyWindow('GFTT Tournament')
        except Exception:
            pass


def _build_grid(
    *,
    input_path: Path,
    dims: Sequence[str],
    smoothing_windows: Sequence[int],
    max_corners: Sequence[int],
    quality_levels: Sequence[float],
    min_distances: Sequence[float],
    use_masking: Sequence[bool],
    output_fourcc: str,
    block_size: int,
) -> list[GFTTSpec]:
    width, height = _get_video_wh(input_path)
    max_dim_full = float('inf')
    max_dim_half = max(1, int(round(max(width, height) / 2.0)))
    dim_map: dict[str, float | int] = {'full': max_dim_full, 'half': int(max_dim_half)}

    specs: list[GFTTSpec] = []
    for dim_key in dims:
        if dim_key not in dim_map:
            raise ValueError(f'Unknown dim option: {dim_key!r} (expected: full|half)')
        for sw in smoothing_windows:
            for mc in max_corners:
                for ql in quality_levels:
                    for md in min_distances:
                        for m in use_masking:
                            specs.append(
                                GFTTSpec(
                                    processing_max_dim=dim_map[dim_key],
                                    smoothing_window=int(sw),
                                    max_corners=int(mc),
                                    quality_level=float(ql),
                                    min_distance=float(md),
                                    block_size=int(block_size),
                                    use_masking=bool(m),
                                    output_fourcc=str(output_fourcc),
                                )
                            )
    return specs


def _render_path_for_spec(*, renders_dir: Path, base: str, spec: GFTTSpec, ext: str) -> Path:
    dim_tag = 'full' if np.isinf(spec.processing_max_dim) else f'max{_float_tag(spec.processing_max_dim)}'
    mask_tag = 'mask' if spec.use_masking else 'nomask'
    name = (
        f'{base}'
        f'__kp-GFTT'
        f'__dim-{dim_tag}'
        f'__corners-{int(spec.max_corners)}'
        f'__q-{_float_tag(spec.quality_level)}'
        f'__dist-{_float_tag(spec.min_distance)}'
        f'__sw-{int(spec.smoothing_window)}'
        f'__{mask_tag}'
        f'__border-black'
        f'{ext}'
    )
    return renders_dir / name


def _precompute(
    *,
    input_path: Path,
    run_dir: Path,
    specs: Sequence[GFTTSpec],
    ext: str,
    show_progress: bool,
    playback: bool,
    tracks_pkl: Path | None,
    yolo_model_path: str | None,
    limit_frames: int | None,
    mask_margin_px: int,
    min_kps_for_mask: int,
    previous_candidates: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    base = vc.sanitize_stem(input_path.stem)
    renders_dir = run_dir / 'renders'
    renders_dir.mkdir(parents=True, exist_ok=True)

    need_mask = any(s.use_masking for s in specs)
    bboxes_by_frame: Mapping[int, Sequence[Sequence[int]]] | None = None
    tracks_path: str | None = None
    if need_mask:
        tracks_pkl_path = vc.ensure_tracks_pkl(
            video_path=input_path,
            output_dir=run_dir,
            tracks_pkl=tracks_pkl,
            yolo_model_path=yolo_model_path,
            limit_frames=limit_frames,
        )
        tracks_path = tracks_pkl_path.as_posix()
        bboxes = vc.load_bboxes_by_frame_from_tracks_pkl(tracks_pkl_path)
        if limit_frames is not None:
            bboxes = {int(k): v for k, v in bboxes.items() if int(k) < int(limit_frames)}
        bboxes_by_frame = bboxes

    prev = dict(previous_candidates or {})
    candidates: dict[str, dict[str, Any]] = {}
    stats = {'rendered': 0, 'skipped': 0, 'failed': 0}

    for i, spec in enumerate(specs, start=1):
        out_path = _render_path_for_spec(renders_dir=renders_dir, base=base, spec=spec, ext=ext)
        cid = _spec_id(spec)
        candidates[cid] = {'id': cid, 'spec': asdict(spec), 'video': out_path.as_posix()}
        if cid in prev and isinstance(prev[cid], dict):
            for k in ('render', 'timing', 'time', 'status'):
                if k in prev[cid] and k not in candidates[cid]:
                    candidates[cid][k] = prev[cid][k]

        if out_path.exists() and out_path.stat().st_size > 0:
            stats['skipped'] += 1
            print(f'[{i:03d}/{len(specs):03d}] SKIP {out_path.name}')
            prev_render = candidates[cid].get('render')
            prev_seconds = 0.0
            prev_status = None
            if isinstance(prev_render, dict):
                prev_seconds = float(prev_render.get('seconds', 0.0) or 0.0)
                prev_status = prev_render.get('status')
            candidates[cid]['render'] = {
                'status': prev_status if prev_status in ('rendered', 'failed') else 'skipped',
                'seconds': prev_seconds,
                'last_seen': time.time(),
                'bytes': int(out_path.stat().st_size),
            }
            continue

        print(f'[{i:03d}/{len(specs):03d}] RENDER {out_path.name}')
        start = time.time()
        try:
            stabilize_once(
                input_path=input_path,
                output_path=out_path,
                spec=spec,
                show_progress=show_progress,
                playback=playback,
                bboxes_by_frame=bboxes_by_frame,
                mask_margin_px=int(mask_margin_px),
                min_kps_for_mask=int(min_kps_for_mask),
            )
            if not out_path.exists() or out_path.stat().st_size <= 0:
                raise RuntimeError('No output written.')
            stats['rendered'] += 1
            seconds = float(time.time() - start)
            candidates[cid]['render'] = {
                'status': 'rendered',
                'seconds': seconds,
                'last_seen': time.time(),
                'bytes': int(out_path.stat().st_size),
            }
            print(f'  -> OK ({seconds:.2f}s)')
        except Exception as e:
            stats['failed'] += 1
            print(f'  -> FAIL {e!r}')
            candidates[cid]['render'] = {'status': 'failed', 'error': repr(e), 'last_seen': time.time()}

    meta = {
        'input_video': input_path.as_posix(),
        'renders_dir': renders_dir.as_posix(),
        'tracks_pkl': tracks_path,
        'stats': stats,
        'count': len(specs),
    }
    return meta, candidates


def _reset_tournament(state: dict[str, Any]) -> None:
    state['tournament'] = {
        'stage': 'prelim',
        'prelim': {'pairs': [], 'winners': {}, 'current': 0},
        'main': {'rounds': [], 'current_round': 0},
        'winner': None,
    }


def _tournament_ids(state: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    t = state.get('tournament', {}) or {}
    prelim = t.get('prelim', {}) or {}
    for pair in prelim.get('pairs', []) or []:
        if isinstance(pair, list) and len(pair) == 2:
            ids.add(str(pair[0]))
            ids.add(str(pair[1]))
    for cid in prelim.get('rest', []) or []:
        ids.add(str(cid))
    main = t.get('main', {}) or {}
    for rnd in main.get('rounds', []) or []:
        for pair in rnd.get('pairs', []) or []:
            if isinstance(pair, list) and len(pair) == 2:
                ids.add(str(pair[0]))
                ids.add(str(pair[1]))
        winners = rnd.get('winners', {}) or {}
        for cid in winners.values():
            ids.add(str(cid))
    winner = t.get('winner')
    if winner:
        ids.add(str(winner))
    return ids


def _state_path(run_dir: Path) -> Path:
    return run_dir / 'tournament_state.json'


def _init_state(
    *,
    input_path: Path,
    run_dir: Path,
    seed: int,
    grid: dict[str, Any],
    precompute_meta: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        'version': 1,
        'created_at': time.time(),
        'seed': int(seed),
        'run_dir': run_dir.as_posix(),
        'input_video': input_path.as_posix(),
        'grid': grid,
        'precompute': precompute_meta,
        'candidates': candidates,
        'tournament': {
            'stage': 'prelim',
            'prelim': {'pairs': [], 'winners': {}, 'current': 0},
            'main': {'rounds': [], 'current_round': 0},
            'winner': None,
        },
    }


def _build_prelim_pairs(*, seed: int, candidate_ids: list[str]) -> tuple[list[list[str]], list[str]]:
    rng = np.random.default_rng(int(seed))
    ids = candidate_ids[:]
    rng.shuffle(ids)

    n = len(ids)
    p = _next_power_of_two_below(n)
    if p == n:
        return [], ids

    elim = n - p
    take = 2 * elim
    prelim_ids = ids[:take]
    rest = ids[take:]
    pairs = [[prelim_ids[i], prelim_ids[i + 1]] for i in range(0, len(prelim_ids), 2)]
    return pairs, rest


def _ensure_bracket(state: dict[str, Any]) -> None:
    t = state['tournament']
    if t['prelim']['pairs']:
        return

    candidate_ids = list(state['candidates'].keys())
    pairs, rest = _build_prelim_pairs(seed=int(state['seed']), candidate_ids=candidate_ids)
    t['prelim']['pairs'] = pairs
    t['prelim']['rest'] = rest
    t['prelim']['current'] = 0
    t['prelim']['winners'] = t['prelim'].get('winners', {})
    if not pairs:
        t['stage'] = 'main'


def _ensure_main_after_prelim(state: dict[str, Any]) -> None:
    t = state['tournament']
    if t['stage'] == 'done':
        return
    if t['main']['rounds']:
        return

    prelim_pairs: list[list[str]] = t['prelim']['pairs']
    winners: dict[str, str] = t['prelim']['winners']
    if prelim_pairs:
        if len(winners) < len(prelim_pairs):
            return
        prelim_winners = [winners[str(i)] for i in range(len(prelim_pairs))]
        participants = prelim_winners + list(t['prelim']['rest'])
    else:
        participants = list(state['candidates'].keys())

    if len(participants) == 1:
        t['winner'] = participants[0]
        t['stage'] = 'done'
        return

    n = len(participants)
    if (n & (n - 1)) != 0:
        raise ValueError(f'Expected power-of-two after prelim; got {n}.')

    rounds = []
    current = participants[:]
    round_idx = 0
    while len(current) > 1:
        pairs = [[current[i], current[i + 1]] for i in range(0, len(current), 2)]
        rounds.append({'round': round_idx, 'pairs': pairs, 'winners': {}, 'current': 0})
        current = ['__TBD__'] * (len(current) // 2)
        round_idx += 1

    t['main']['rounds'] = rounds
    t['main']['current_round'] = 0
    t['stage'] = 'main'


def _run_tournament(*, state: dict[str, Any], state_path: Path, viewer_scale: float) -> None:
    _ensure_bracket(state)
    _ensure_main_after_prelim(state)
    seed = int(state['seed'])
    t = state['tournament']
    viewer_scale = float(viewer_scale)
    if viewer_scale <= 0:
        viewer_scale = 1.0

    def get_video(cid: str) -> Path:
        return Path(state['candidates'][cid]['video'])

    while True:
        _safe_write_json(state_path, state)
        stage = t['stage']
        if stage == 'done':
            return

        if stage == 'prelim':
            pairs: list[list[str]] = t['prelim']['pairs']
            current = int(t['prelim']['current'])
            if current >= len(pairs):
                _ensure_main_after_prelim(state)
                continue

            if str(current) in t['prelim']['winners']:
                t['prelim']['current'] = current + 1
                continue

            a, b = pairs[current]
            match_key = f'prelim:{current}'
            left_id, right_id = _deterministic_lr(seed, match_key, a, b)
            title = f'Prelim {current + 1}/{len(pairs)} (A vs B)'
            choice = _play_match(
                left_video=get_video(left_id),
                right_video=get_video(right_id),
                title=title,
                viewer_scale=viewer_scale,
            )
            if choice is None:
                _safe_write_json(state_path, state)
                return
            winner = left_id if choice == 0 else right_id
            t['prelim']['winners'][str(current)] = winner
            t['prelim']['current'] = current + 1
            continue

        if stage == 'main':
            rounds: list[dict[str, Any]] = t['main']['rounds']
            r_idx = int(t['main']['current_round'])
            if r_idx >= len(rounds):
                t['stage'] = 'done'
                return
            rnd = rounds[r_idx]
            pairs: list[list[str]] = rnd['pairs']
            current = int(rnd['current'])
            winners: dict[str, str] = rnd['winners']

            if current >= len(pairs):
                winners_in_order = [winners[str(i)] for i in range(len(pairs))]
                if len(winners_in_order) == 1:
                    t['winner'] = winners_in_order[0]
                    t['stage'] = 'done'
                    _safe_write_json(state_path, state)
                    return

                next_round = rounds[r_idx + 1]
                next_round['pairs'] = [
                    [winners_in_order[i], winners_in_order[i + 1]] for i in range(0, len(winners_in_order), 2)
                ]
                next_round['current'] = 0
                t['main']['current_round'] = r_idx + 1
                continue

            if str(current) in winners:
                rnd['current'] = current + 1
                continue

            a, b = pairs[current]
            match_key = f'main:{r_idx}:{current}'
            left_id, right_id = _deterministic_lr(seed, match_key, a, b)
            title = f'Round {r_idx + 1}/{len(rounds)} Match {current + 1}/{len(pairs)} (A vs B)'
            choice = _play_match(
                left_video=get_video(left_id),
                right_video=get_video(right_id),
                title=title,
                viewer_scale=viewer_scale,
            )
            if choice is None:
                _safe_write_json(state_path, state)
                return
            winner = left_id if choice == 0 else right_id
            winners[str(current)] = winner
            rnd['current'] = current + 1
            continue

        raise ValueError(f'Unknown stage: {stage!r}')


def _export_results(*, state: dict[str, Any], out_path: Path) -> None:
    t = state['tournament']
    winner_id = t.get('winner')
    wins_losses: dict[str, dict[str, int]] = {cid: {'wins': 0, 'losses': 0} for cid in state.get('candidates', {})}

    prelim = t.get('prelim', {})
    prelim_pairs: list[list[str]] = prelim.get('pairs', []) or []
    prelim_winners: dict[str, str] = prelim.get('winners', {}) or {}
    for i, pair in enumerate(prelim_pairs):
        w = prelim_winners.get(str(i))
        if not w:
            continue
        a, b = pair
        loser = b if w == a else a
        if w in wins_losses:
            wins_losses[w]['wins'] += 1
        if loser in wins_losses:
            wins_losses[loser]['losses'] += 1

    main = t.get('main', {})
    rounds: list[dict[str, Any]] = main.get('rounds', []) or []
    for rnd in rounds:
        pairs: list[list[str]] = rnd.get('pairs', []) or []
        winners: dict[str, str] = rnd.get('winners', {}) or {}
        for i, pair in enumerate(pairs):
            w = winners.get(str(i))
            if not w:
                continue
            a, b = pair
            loser = b if w == a else a
            if w in wins_losses:
                wins_losses[w]['wins'] += 1
            if loser in wins_losses:
                wins_losses[loser]['losses'] += 1

    ranking = sorted(
        [{'id': cid, **wl, 'spec': state['candidates'][cid]['spec']} for cid, wl in wins_losses.items()],
        key=lambda r: (-int(r['wins']), int(r['losses']), r['id']),
    )
    summary: dict[str, Any] = {
        'winner_id': winner_id,
        'winner': state['candidates'].get(winner_id) if winner_id else None,
        'input_video': state['input_video'],
        'seed': state['seed'],
        'grid': state['grid'],
        'win_loss_by_id': wins_losses,
        'ranking': ranking,
    }
    _safe_write_json(out_path, summary)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--mode', choices=['all', 'precompute', 'tournament'], default='all')
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--ext', default='.mp4')
    ap.add_argument('--fourcc', default='mp4v')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--viewer-scale', type=float, default=0.5, help='Scale factor for tournament viewer (default: 0.5).')

    ap.add_argument('--dims', nargs='*', default=['full', 'half'])
    ap.add_argument('--smoothing-windows', nargs='*', type=int, default=[5, 10, 20])
    ap.add_argument('--max-corners', nargs='*', type=int, default=[200, 500, 1000])
    ap.add_argument('--quality-levels', nargs='*', type=float, default=[0.01, 0.03])
    ap.add_argument('--min-distances', nargs='*', type=float, default=[10.0, 30.0])
    ap.add_argument('--block-size', type=int, default=3)
    ap.add_argument('--masking', nargs='*', default=['on', 'off'])

    ap.add_argument('--tracks-pkl', type=str, default=None)
    ap.add_argument('--yolo-model-path', type=str, default=None)
    ap.add_argument('--limit-frames', type=int, default=None)
    ap.add_argument('--mask-margin-px', type=int, default=20)
    ap.add_argument('--min-kps-for-mask', type=int, default=30)

    ap.add_argument('--playback', action='store_true')
    ap.add_argument('--no-progress', action='store_true')
    args = ap.parse_args()

    input_path = Path(args.video).expanduser().resolve()
    if not input_path.exists():
        print(f'[ERROR] Not found: {input_path}', file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else _default_run_dir(input_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    ext = args.ext if args.ext.startswith('.') else f'.{args.ext}'
    use_masking = [m.lower() in ('on', 'true', '1', 'yes') for m in args.masking]
    grid = {
        'dims': list(args.dims),
        'smoothing_windows': list(args.smoothing_windows),
        'max_corners': list(args.max_corners),
        'quality_levels': list(args.quality_levels),
        'min_distances': list(args.min_distances),
        'block_size': int(args.block_size),
        'masking': list(args.masking),
        'ext': ext,
        'fourcc': args.fourcc,
    }

    specs = _build_grid(
        input_path=input_path,
        dims=args.dims,
        smoothing_windows=args.smoothing_windows,
        max_corners=args.max_corners,
        quality_levels=args.quality_levels,
        min_distances=args.min_distances,
        use_masking=use_masking,
        output_fourcc=args.fourcc,
        block_size=args.block_size,
    )
    total = len(specs)
    print(
        f'Grid size: {total} '
        f'(dims={len(args.dims)} corners={len(args.max_corners)} q={len(args.quality_levels)} '
        f'dist={len(args.min_distances)} sw={len(args.smoothing_windows)} masking={len(use_masking)})'
    )

    state_path = _state_path(run_dir)
    state = _load_json(state_path)

    if args.mode in ('all', 'precompute'):
        tracks_pkl = Path(args.tracks_pkl).expanduser().resolve() if args.tracks_pkl else None
        prev_candidates = state.get('candidates') if isinstance(state, dict) else None
        pre_meta, candidates = _precompute(
            input_path=input_path,
            run_dir=run_dir,
            specs=specs,
            ext=ext,
            show_progress=not args.no_progress,
            playback=bool(args.playback),
            tracks_pkl=tracks_pkl,
            yolo_model_path=args.yolo_model_path,
            limit_frames=args.limit_frames,
            mask_margin_px=int(args.mask_margin_px),
            min_kps_for_mask=int(args.min_kps_for_mask),
            previous_candidates=prev_candidates,
        )
        if state is None:
            state = _init_state(
                input_path=input_path,
                run_dir=run_dir,
                seed=int(args.seed),
                grid=grid,
                precompute_meta=pre_meta,
                candidates=candidates,
            )
        else:
            state['precompute'] = pre_meta
            state['candidates'] = candidates
            state['grid'] = grid
            old_ids = set((prev_candidates or {}).keys()) if isinstance(prev_candidates, dict) else set()
            new_ids = set(candidates.keys())
            if old_ids and old_ids != new_ids:
                _reset_tournament(state)
            else:
                bad = _tournament_ids(state) - new_ids
                if bad:
                    _reset_tournament(state)
        _safe_write_json(state_path, state)

    if args.mode in ('all', 'tournament'):
        if state is None:
            print(f'[ERROR] No state found at {state_path}. Run with --mode precompute first.', file=sys.stderr)
            return 2
        if _tournament_ids(state) - set(state.get('candidates', {}).keys()):
            _reset_tournament(state)
        try:
            _run_tournament(state=state, state_path=state_path, viewer_scale=float(args.viewer_scale))
        finally:
            _safe_write_json(state_path, state)

        if state.get('tournament', {}).get('stage') == 'done':
            out = run_dir / 'results.json'
            _export_results(state=state, out_path=out)
            winner_id = state['tournament'].get('winner')
            if winner_id:
                winner = state['candidates'][winner_id]
                print('Winner:')
                print(json.dumps(winner, indent=2, sort_keys=True))
            print(f'Wrote {out.as_posix()}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
