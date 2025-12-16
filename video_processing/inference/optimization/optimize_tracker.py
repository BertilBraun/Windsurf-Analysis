from __future__ import annotations

import sys
import math
import optuna
import argparse

from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from server.inference.src.tracking.ilp_tracker import ILPTracker
from server.inference.src.tracking.iterative_ilp_tracker import IterativeILPTracker
from server.inference.src.util.video_io import get_video_properties
from server.inference.src.tracking.preprocessing.preprocessor import TrackPreProcessor
from server.inference.src.tracking.discrete_opt_tracker import DiscreteOptTracker
from server.inference.src.common_types import Track
from server.inference.src.visualization.stabilize import compute_stabilization_transforms_gmc
from server.inference.optimization.optimization_util import (
    PairwiseScores,
    optimize,
    each_golden,
    build_assignment_from_tracks,
    build_assignment_from_metadata,
    pairwise_scores,
)


def _flatten_tracks(tracks: List[Track]) -> List[Track]:
    flat_tracks: List[Track] = []
    for track in tracks:
        for detection in track.sorted_detections:
            flat_tracks.append(Track(track_id=len(flat_tracks) + 1, sorted_detections=[detection]))
    return flat_tracks


def _evaluate_preprocessor(params: Dict[str, Any], golden_dir: Path) -> float:
    total_tracks = 0
    num_videos = 0
    total_impure = 0

    for tracks, meta in each_golden(golden_dir):
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
            total_impure += 1
            num_videos += 1
            total_tracks += len(pred_tracks)
            continue

        gold_ids_by_pred: Dict[int, set[int]] = defaultdict(set)
        for det_key, pred_tid in pred_assign.items():
            gold_tid = gold_assign[det_key]
            gold_ids_by_pred[int(pred_tid)].add(int(gold_tid))

        impure = sum(1 for s in gold_ids_by_pred.values() if len(s) > 1)
        total_impure += impure
        total_tracks += len(gold_ids_by_pred)
        num_videos += 1

    if num_videos == 0:
        return -1e9

    avg_tracks = total_tracks / max(1, num_videos)
    if total_impure > 0:
        score = -100.0 * float(total_impure) - avg_tracks
    else:
        score = 100.0 - avg_tracks
    return float(score)


def _run_opt_preprocessor(args) -> Tuple[float, Dict[str, Any]]:
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
        score = _evaluate_preprocessor(params, args.golden_dir)
        return float(score)

    study = optimize(objective, direction='maximize', trials=args.trials, seed=args.seed)

    best_params = dict(study.best_trial.params)
    best_score = _evaluate_preprocessor(best_params, args.golden_dir)
    return float(best_score), dict(best_params)


def _evaluate_discrete(params: Dict[str, Any], golden_dir: Path) -> Tuple[float, List[Tuple[str, PairwiseScores]]]:
    metrics_list: List[Tuple[str, PairwiseScores]] = []

    for tracks, meta in each_golden(golden_dir):
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
        metrics_list.append((video_path.name, s))

    if not metrics_list:
        return float('nan'), []

    avg_f1 = sum(s.f1 for _, s in metrics_list) / len(metrics_list)
    return float(avg_f1), metrics_list


def _run_opt_discrete(args) -> Tuple[float, Dict[str, Any]]:
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
        score, _ = _evaluate_discrete(params, args.golden_dir)
        return -1.0 if math.isnan(score) else float(score)

    study = optimize(objective, direction='maximize', trials=args.trials, seed=args.seed)

    best_score = float(study.best_value)
    best_params = dict(study.best_trial.params)
    best_params['long_link_cost_appearance_window_radius'] = 999999
    return float(best_score), dict(best_params)


# ─────────────────────────────── BoT-SORT mode ─────────────────────────────── #


def _evaluate_iter_ilp(params: Dict[str, Any], golden_dir: Path) -> float:
    metrics: List[Tuple[PairwiseScores, float]] = []

    for tracks, meta in each_golden(golden_dir):
        video_path = Path(meta.input_video_path)
        video_props = get_video_properties(video_path)
        input_tracks = _flatten_tracks(tracks)
        transforms = compute_stabilization_transforms_gmc(video_path.as_posix())

        preprocessed_tracks = TrackPreProcessor().track(input_tracks, video_props, transforms)

        # tracker = IterativeILPTracker(video_path.as_posix(), **params)
        tracker = ILPTracker(video_path.as_posix(), **params)

        # iterative_ilp_tracks, iterations = tracker._internal_track_with_iteration_returned(
        #     preprocessed_tracks, video_props, transforms
        # )
        iterative_ilp_tracks = tracker.track(preprocessed_tracks, video_props, transforms)

        gold_assign = build_assignment_from_metadata(meta)
        pred_assign = build_assignment_from_tracks(iterative_ilp_tracks)

        s = pairwise_scores(gold_assign, pred_assign)
        # metrics.append((s, iterations))
        metrics.append((s, 0))

    if not metrics:
        return float('nan')

    average_iterations = sum(iterations for _, iterations in metrics) / len(metrics)
    print(f'Average iterations: {average_iterations}')
    weighted_avg_f1 = sum(s.f1 * s.num_detections for s, _ in metrics) / sum(s.num_detections for s, _ in metrics)
    print(f'Weighted avg F1: {weighted_avg_f1}')
    return weighted_avg_f1 - average_iterations / 100


def _run_iter_ilp(args) -> Tuple[float, Dict[str, Any]]:
    def objective(trial: optuna.trial.Trial) -> float:
        params: Dict[str, Any] = {
            # NOTE: All commented out params are not used with only one optimization iteration
            'w_start': trial.suggest_float('w_start', 0.5, 100.0),
            # 'start_cost_mode': trial.suggest_categorical('start_cost_mode', ['linear', 'geo']),
            # 'start_cost_growth': trial.suggest_float('start_cost_growth', 0.0, 10.0),
            # 'start_cost_max': trial.suggest_float('start_cost_max', 0.0, 100.0),
            'w_motion': trial.suggest_float('w_motion', 0.0, 10.0),
            'w_appearance': trial.suggest_float('w_appearance', 0.0, 10.0),
            'w_gap': trial.suggest_float('w_gap', 0.0, 10.0),
            'p_miss': trial.suggest_float('p_miss', 0.8, 1.0),
            # 'max_detections_to_compare': trial.suggest_int('max_detections_to_compare', 1, 4),
            # 'use_position_only': trial.suggest_categorical('use_position_only', [True, False]),
            # 'max_optimization_iterations': trial.suggest_int('max_optimization_iterations', 2, 5),
            # 'internal_split_gap_frames': trial.suggest_int('internal_split_gap_frames', 1, 10),
            # 'motion_split_nll_max': trial.suggest_float('motion_split_nll_max', 0.0, 10.0),
            # 'appearance_split_nll_max': trial.suggest_float('appearance_split_nll_max', 0.0, 10.0),
            # 'appearance_split_window': trial.suggest_int('appearance_split_window', 1, 10),
            # 'max_splits_per_track': trial.suggest_int('max_splits_per_track', 1, 11, step=2),
            'appearance_similarity_gamma': trial.suggest_float('appearance_similarity_gamma', 2.0, 15.0),
        }
        score = _evaluate_iter_ilp(params, args.golden_dir)
        return -1.0 if math.isnan(score) else float(score)

    study = optimize(objective, direction='maximize', trials=args.trials, seed=args.seed)
    best_params = dict(study.best_trial.params)
    best_score = _evaluate_iter_ilp(best_params, args.golden_dir)
    return float(best_score), dict(best_params)


# ─────────────────────────────────── CLI ───────────────────────────────────── #


def main() -> None:
    parser = argparse.ArgumentParser(description='Unified optimizer for Preprocessor, Discrete ILP, and BoT-SORT.')
    parser.add_argument('--mode', type=str, required=True, choices=['preprocessor', 'ilp', 'iter_ilp', 'botsort'])
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory with *.golden.tracks.pkl files')
    parser.add_argument('--trials', type=int, default=200, help='Number of Optuna trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

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
