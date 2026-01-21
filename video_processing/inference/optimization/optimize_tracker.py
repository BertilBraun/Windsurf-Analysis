from __future__ import annotations

import sys
import math
import optuna
import argparse
import os
from functools import cache
from time import perf_counter

from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from video_processing.inference.src.tracking.ilp_tracker import ILPTracker
from video_processing.inference.src.tracking.iterative_ilp_tracker import IterativeILPTracker
from video_processing.inference.src.util.video_io import get_video_properties
from video_processing.inference.src.tracking.preprocessing.preprocessor import TrackPreProcessor
from video_processing.inference.src.tracking.discrete_opt_tracker import DiscreteOptTracker
from video_processing.inference.src.common_types import Track
from video_processing.inference.src.visualization.stabilize import compute_stabilization_transforms_gmc
from video_processing.inference.optimization.optimization_util import (
    PairwiseScores,
    optimize,
    build_assignment_from_tracks,
    build_assignment_from_metadata,
    pairwise_scores,
    list_golden_paths,
    load_full_tracks,
    SharedQueueWorkerPool,
)


def _flatten_tracks(tracks: List[Track]) -> List[Track]:
    flat_tracks: List[Track] = []
    for track in tracks:
        for detection in track.sorted_detections:
            flat_tracks.append(Track(track_id=len(flat_tracks) + 1, sorted_detections=[detection]))
    return flat_tracks


def _resolve_workers(workers: int) -> int:
    return max(1, (int(os.cpu_count() or 1) if workers <= 0 else workers))


def _preprocessor_one(golden_path: str, params: Dict[str, Any]) -> tuple[int, int]:
    # Worker: returns (impure_count, num_pred_tracks)
    tracks, meta = load_full_tracks(Path(golden_path))
    video_path = Path(meta.input_video_path)
    props = get_video_properties(video_path)
    transforms = compute_stabilization_transforms_gmc(video_path.as_posix())

    pre = TrackPreProcessor(
        appearance_probability_strict=float(params['appearance_strict']),
        appearance_probability_loose=float(params['appearance_loose']),
        motion_probability_strict=float(params['motion_strict']),
        motion_probability_loose=float(params['motion_loose']),
        max_frame_distance=int(params['max_frame_distance']),
        ema_alpha=float(params['ema_alpha']),
        debug_video_path=video_path.as_posix(),
    )
    initial_tracks = _flatten_tracks(tracks)
    pred_tracks = pre.track(initial_tracks, props, transforms)

    gold_assign = build_assignment_from_metadata(meta)
    pred_assign = build_assignment_from_tracks(pred_tracks)

    if set(gold_assign.keys()) != set(pred_assign.keys()):
        # Count as impure and use predicted track count for fragmentation term.
        return 1, int(len(pred_tracks))

    gold_ids_by_pred: Dict[int, set[int]] = defaultdict(set)
    for det_key, pred_tid in pred_assign.items():
        gold_tid = gold_assign[det_key]
        gold_ids_by_pred[int(pred_tid)].add(int(gold_tid))

    impure = sum(1 for s in gold_ids_by_pred.values() if len(s) > 1)
    return int(impure), int(len(gold_ids_by_pred))


def _evaluate_preprocessor(
    params: Dict[str, Any],
    golden_paths: List[Path],
    *,
    pool: SharedQueueWorkerPool,
) -> float:
    if not golden_paths:
        return -1e9

    total_tracks = 0
    total_impure = 0

    results = pool.map_paths([p.as_posix() for p in golden_paths], params)
    for r in results:
        impure, n_tracks = r  # type: ignore[misc]
        total_impure += int(impure)
        total_tracks += int(n_tracks)

    avg_tracks = total_tracks / max(1, len(golden_paths))
    if total_impure > 0:
        score = -100.0 * float(total_impure) - avg_tracks
    else:
        score = 100.0 - avg_tracks
    return float(score)


def _run_opt_preprocessor(args) -> Tuple[float, Dict[str, Any]]:
    golden_paths = list_golden_paths(args.golden_dir)
    workers = min(_resolve_workers(args.workers), max(1, len(golden_paths)))
    with SharedQueueWorkerPool(worker_fn=_preprocessor_one, workers=workers) as pool:

        def objective(trial: optuna.trial.Trial) -> float:
            params = {
                'appearance_strict': trial.suggest_float('appearance_strict', 0.70, 1.0),
                'appearance_loose': trial.suggest_float('appearance_loose', 0.05, 1.0),
                'motion_strict': trial.suggest_float('motion_strict', 0.70, 1.0),
                'motion_loose': trial.suggest_float('motion_loose', 0.05, 1.0),
                'max_frame_distance': trial.suggest_int('max_frame_distance', 0, 10),
                'ema_alpha': trial.suggest_float('ema_alpha', 0.0, 1.0),
            }
            if params['appearance_strict'] <= params['appearance_loose']:
                return -math.inf
            if params['motion_strict'] <= params['motion_loose']:
                return -math.inf
            score = _evaluate_preprocessor(params, golden_paths, pool=pool)
            return float(score)

        study = optimize(objective, direction='maximize', trials=args.trials, seed=args.seed)

        best_params = dict(study.best_trial.params)
        best_score = _evaluate_preprocessor(best_params, golden_paths, pool=pool)
        return float(best_score), dict(best_params)


def _discrete_one(golden_path: str, params: Dict[str, Any]) -> tuple[str, PairwiseScores]:
    tracks, meta = load_full_tracks(Path(golden_path))
    video_path = Path(meta.input_video_path)
    video_props = get_video_properties(video_path)

    input_tracks = _flatten_tracks(tracks)
    transforms = compute_stabilization_transforms_gmc(video_path.as_posix())
    tracker = DiscreteOptTracker(
        short_min_link_iou=float(params['short_min_link_iou']),
        short_min_link_cos=float(params['short_min_link_cos']),
        short_w_link_iou=float(params['short_w_link_iou']),
        short_w_link_app=float(params['short_w_link_app']),
        short_w_link_gap=float(params['short_w_link_gap']),
        short_link_cost_appearance_window_radius=int(params['short_link_cost_appearance_window_radius']),
        long_min_link_iou=float(params['long_min_link_iou']),
        long_min_link_cos=float(params['long_min_link_cos']),
        long_w_link_iou=float(params['long_w_link_iou']),
        long_w_link_app=float(params['long_w_link_app']),
        long_w_link_gap=float(params['long_w_link_gap']),
        long_link_cost_appearance_window_radius=int(params['long_link_cost_appearance_window_radius']),
        w_start=float(params['w_start']),
    )

    pred_tracks = tracker.track(input_tracks, video_props, transforms)
    gold_assign = build_assignment_from_metadata(meta)
    pred_assign = build_assignment_from_tracks(pred_tracks)
    s = pairwise_scores(gold_assign, pred_assign)
    return str(video_path.name), s


def _evaluate_discrete(
    params: Dict[str, Any],
    golden_paths: List[Path],
    *,
    pool: SharedQueueWorkerPool[Tuple[str, PairwiseScores]],
) -> Tuple[float, List[Tuple[str, PairwiseScores]]]:
    if not golden_paths:
        return float('nan'), []

    metrics_list = pool.map_paths([p.as_posix() for p in golden_paths], params)
    avg_f1 = sum(s.f1 for _, s in metrics_list) / len(metrics_list)
    return float(avg_f1), metrics_list


def _run_opt_discrete(args) -> Tuple[float, Dict[str, Any]]:
    golden_paths = list_golden_paths(args.golden_dir)
    workers = min(_resolve_workers(args.workers), max(1, len(golden_paths)))
    with SharedQueueWorkerPool(worker_fn=_discrete_one, workers=workers) as pool:

        def objective(trial: optuna.trial.Trial) -> float:
            params: Dict[str, Any] = {
                'w_start': trial.suggest_float('w_start', 0.5, 10.0),
                'short_min_link_iou': trial.suggest_float('short_min_link_iou', 0.0, 0.6),
                'short_min_link_cos': trial.suggest_float('short_min_link_cos', 0.0, 0.9),
                'short_w_link_iou': trial.suggest_float('short_w_link_iou', 0.0, 1.0),
                'short_w_link_app': trial.suggest_float('short_w_link_app', 0.2, 3.0),
                'short_w_link_gap': trial.suggest_float('short_w_link_gap', 0.0, 0.2),
                'short_link_cost_appearance_window_radius': trial.suggest_int(
                    'short_link_cost_appearance_window_radius', 0, 20
                ),
                'long_min_link_iou': trial.suggest_float('long_min_link_iou', 0.0, 0.5),
                'long_min_link_cos': trial.suggest_float('long_min_link_cos', 0.3, 0.95),
                'long_w_link_iou': trial.suggest_float('long_w_link_iou', 0.0, 0.6),
                'long_w_link_app': trial.suggest_float('long_w_link_app', 0.2, 4.0),
                'long_w_link_gap': trial.suggest_float('long_w_link_gap', 0.0, 4.0),
                'long_link_cost_appearance_window_radius': 999999,
            }
            score, _ = _evaluate_discrete(params, golden_paths, pool=pool)
            return -1.0 if math.isnan(score) else float(score)

        study = optimize(objective, direction='maximize', trials=args.trials, seed=args.seed)

        best_score = float(study.best_value)
        best_params = dict(study.best_trial.params)
        best_params['long_link_cost_appearance_window_radius'] = 999999
        return float(best_score), dict(best_params)


# ─────────────────────────────── BoT-SORT mode ─────────────────────────────── #


def _iter_ilp_one(golden_path: str, params: Dict[str, Any]) -> tuple[PairwiseScores, float]:
    video_path, video_props, transforms, preprocessed_tracks, gold_assign = _iter_ilp_case(golden_path)

    tracker = ILPTracker(video_path, **params)
    pred_tracks = tracker.track(preprocessed_tracks, video_props, transforms)
    pred_assign = build_assignment_from_tracks(pred_tracks)
    s = pairwise_scores(gold_assign, pred_assign)
    return s, 0.0


@cache
def _iter_ilp_case(golden_path: str) -> tuple[str, Any, Any, List[Track], Dict[Any, int]]:
    # Cache all trial-invariant work per golden file within the current process.
    t0 = perf_counter()
    tracks, meta = load_full_tracks(Path(golden_path))
    video_path = Path(meta.input_video_path).as_posix()
    video_props = get_video_properties(video_path)
    transforms = compute_stabilization_transforms_gmc(video_path)

    input_tracks = _flatten_tracks(tracks)
    preprocessed_tracks = TrackPreProcessor().track(input_tracks, video_props, transforms)
    gold_assign = build_assignment_from_metadata(meta)
    if os.getenv('WINDSURF_ITER_ILP_PROFILE'):
        dt = perf_counter() - t0
        print(f'[iter_ilp_case] prepared {Path(golden_path).name} in {dt:.3f}s', file=sys.stderr, flush=True)
    return str(video_path), video_props, transforms, preprocessed_tracks, gold_assign


def _evaluate_iter_ilp(
    params: Dict[str, Any],
    golden_paths: List[Path],
    *,
    pool: SharedQueueWorkerPool[Tuple[PairwiseScores, float]],
) -> float:
    if not golden_paths:
        return float('nan')

    metrics = pool.map_paths([p.as_posix() for p in golden_paths], params)
    average_iterations = sum(iterations for _, iterations in metrics) / len(metrics)
    weighted_avg_f1 = sum(s.f1 * s.num_detections for s, _ in metrics) / sum(s.num_detections for s, _ in metrics)
    return float(weighted_avg_f1 - average_iterations / 100)


def _run_iter_ilp(args) -> Tuple[float, Dict[str, Any]]:
    golden_paths = list_golden_paths(args.golden_dir)
    workers = min(_resolve_workers(args.workers), max(1, len(golden_paths)))

    with SharedQueueWorkerPool(worker_fn=_iter_ilp_one, workers=workers) as pool:

        def objective(trial: optuna.trial.Trial) -> float:
            params: Dict[str, Any] = {
                'w_start': trial.suggest_float('w_start', 0.5, 100.0),
                'w_motion': trial.suggest_float('w_motion', 0.0, 10.0),
                'w_appearance': trial.suggest_float('w_appearance', 0.0, 10.0),
                'w_gap': trial.suggest_float('w_gap', 0.0, 10.0),
                'p_miss': trial.suggest_float('p_miss', 0.8, 1.0),
                'appearance_similarity_gamma': trial.suggest_float('appearance_similarity_gamma', 2.0, 15.0),
            }
            score = _evaluate_iter_ilp(params, golden_paths, pool=pool)
            return -1.0 if math.isnan(score) else float(score)

        study = optimize(objective, direction='maximize', trials=args.trials, seed=args.seed)
        best_params = dict(study.best_trial.params)
        best_score = _evaluate_iter_ilp(best_params, golden_paths, pool=pool)
        return float(best_score), dict(best_params)


# ─────────────────────────────────── CLI ───────────────────────────────────── #


def main() -> None:
    parser = argparse.ArgumentParser(description='Unified optimizer for Preprocessor, Discrete ILP, and BoT-SORT.')
    parser.add_argument('--mode', type=str, required=True, choices=['preprocessor', 'ilp', 'iter_ilp', 'botsort'])
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory with *.golden.tracks.pkl files')
    parser.add_argument('--trials', type=int, default=200, help='Number of Optuna trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workers', type=int, default=0, help='Parallel workers (0 = cpu_count)')

    args = parser.parse_args()

    mode = args.mode
    best_score: float
    best_params: Dict[str, Any]

    if mode == 'preprocessor':
        best_score, best_params = _run_opt_preprocessor(args)
    elif mode == 'ilp':
        best_score, best_params = _run_opt_discrete(args)
    elif mode == 'iter_ilp':
        best_score, best_params = _run_iter_ilp(args)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    if not best_params:
        return

    print('\nBest params (score = %.4f):' % float(best_score))
    for k in sorted(best_params.keys()):
        print(f'  {k}: {best_params[k]}')


if __name__ == '__main__':
    main()
