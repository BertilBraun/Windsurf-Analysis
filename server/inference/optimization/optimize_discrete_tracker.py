from __future__ import annotations

import sys
import json
import math
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import optuna


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from server.inference.src.util.video_io import get_video_properties
from server.inference.src.tracking.detector import SurferDetector
from server.inference.src.tracking.preprocessing.preprocessor import Preprocessor
from server.inference.src.tracking.discrete_opt_tracker import DiscreteILPTracker
from server.inference.src.common_types import Detection, Track
from server.inference.src.player.core.player_state import Metadata, TrackLite, DetectionLite
from server.inference.src.settings import YOLO_MODEL_PATH


def _detections_to_initial_tracks(detections: List[Detection]) -> List[Track]:
    return [Track(track_id=i + 1, sorted_detections=[det]) for i, det in enumerate(detections)]


def _load_golden(path: Path) -> Metadata:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, Metadata):
        raise TypeError('Golden file does not contain Metadata')
    return data


def _build_assignment_from_tracks(tracks: List[Track]) -> Dict[Tuple[int, int, int, int, int], int]:
    assignment: Dict[Tuple[int, int, int, int, int], int] = {}
    for t in tracks:
        for det in t.sorted_detections:
            k = (int(det.frame_idx), int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2))
            assignment[k] = int(t.track_id)
    return assignment


def _build_assignment_from_metadata(meta: Metadata) -> Dict[Tuple[int, int, int, int, int], int]:
    assignment: Dict[Tuple[int, int, int, int, int], int] = {}
    for t in meta.tracks:
        assert isinstance(t, TrackLite)
        for det in t.detections:
            assert isinstance(det, DetectionLite)
            x1, y1, x2, y2 = det.bbox
            k = (int(det.frame_idx), int(x1), int(y1), int(x2), int(y2))
            assignment[k] = int(t.track_id)
    return assignment


def _pairwise_scores(gold: Dict, pred: Dict) -> Dict[str, float]:
    assert set(gold.keys()) == set(pred.keys()), 'Gold and pred keys must be the same'
    keys = sorted(gold.keys())
    n = len(keys)
    if n < 2:
        return {
            'num_detections': float(n),
            'pairs': 0.0,
            'pairwise_precision': 1.0,
            'pairwise_recall': 1.0,
            'pairwise_f1': 1.0,
            'rand_index': 1.0,
            'jaccard_same': 1.0,
        }

    tp = tn = fp = fn = 0
    for i in range(n):
        gi = gold[keys[i]]
        pi = pred[keys[i]]
        for j in range(i + 1, n):
            gj = gold[keys[j]]
            pj = pred[keys[j]]
            gold_same = gi == gj
            pred_same = pi == pj
            if gold_same and pred_same:
                tp += 1
            elif (not gold_same) and (not pred_same):
                tn += 1
            elif (not gold_same) and pred_same:
                fp += 1
            elif gold_same and (not pred_same):
                fn += 1

    pairs = tp + tn + fp + fn
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 1.0
    rand = (tp + tn) / pairs if pairs else 1.0
    jaccard_same = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    return {
        'num_detections': float(n),
        'pairs': float(pairs),
        'pairwise_precision': prec,
        'pairwise_recall': rec,
        'pairwise_f1': f1,
        'rand_index': rand,
        'jaccard_same': jaccard_same,
    }


class DetectionCache:
    def __init__(self) -> None:
        self._detector = SurferDetector(yolo_model_path=YOLO_MODEL_PATH)
        self._detections_by_video: Dict[Path, List[Detection]] = {}
        self._preprocessed_tracks_by_video: Dict[Path, Tuple[List[Track], Any]] = {}

    def get_preprocessed_tracks(self, video_path: Path) -> Tuple[List[Track], Any]:
        if video_path in self._preprocessed_tracks_by_video:
            return self._preprocessed_tracks_by_video[video_path]

        video_props = get_video_properties(video_path)
        dets = self._detections_by_video.get(video_path)
        if dets is None:
            dets = self._detector.run_object_detection_on_video(video_path)
            self._detections_by_video[video_path] = dets

        initial_tracks = _detections_to_initial_tracks(dets)
        tracks = Preprocessor().track(initial_tracks, video_props)
        self._preprocessed_tracks_by_video[video_path] = (tracks, video_props)
        return tracks, video_props


def _evaluate_params_on_goldens(
    params: Dict[str, Any],
    golden_paths: List[Path],
    cache: DetectionCache,
) -> Tuple[float, List[Tuple[str, Dict[str, float]]]]:
    metrics_list: List[Tuple[str, Dict[str, float]]] = []

    for golden_path in golden_paths:
        gold_meta = _load_golden(golden_path)
        video_path = Path(gold_meta.input_video_path)

        pred_input_tracks, video_props = cache.get_preprocessed_tracks(video_path)
        tracker = DiscreteILPTracker(
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

        pred_tracks = tracker.track(list(pred_input_tracks), video_props)

        gold_assign = _build_assignment_from_metadata(gold_meta)
        pred_assign = _build_assignment_from_tracks(pred_tracks)
        if set(gold_assign.keys()) != set(pred_assign.keys()):
            # Skip if detections differ
            print(f'Detections differ for {golden_path.name}')
            continue
        s = _pairwise_scores(gold_assign, pred_assign)
        metrics_list.append((golden_path.name, s))

    if not metrics_list:
        return float('nan'), []

    avg_f1 = sum(s['pairwise_f1'] for _, s in metrics_list) / len(metrics_list)
    return avg_f1, metrics_list


def main() -> None:
    parser = argparse.ArgumentParser(description='Bayesian optimization for Discrete ILP tracker.')
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory with *.golden.tracks.pkl files')
    parser.add_argument('--trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--metric', type=str, default='pairwise_f1', choices=['pairwise_f1'], help='Optimization metric'
    )
    parser.add_argument('--report-json', type=str, default='', help='Optional path to write detailed results JSON')

    args = parser.parse_args()

    # Seed Optuna's sampler for reproducibility
    golden_dir = Path(args.golden_dir)
    if not golden_dir.exists():
        print(f'Golden dir not found: {golden_dir}')
        return

    golden_paths = sorted(golden_dir.glob('*.golden.tracks.pkl'))
    if not golden_paths:
        print('No golden files found in the specified directory.')
        return

    cache = DetectionCache()

    best_score = -1.0
    best_params: Optional[Dict[str, Any]] = None
    all_results: List[Dict[str, Any]] = []

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
        score, _ = _evaluate_params_on_goldens(params, golden_paths, cache)
        return -1.0 if math.isnan(score) else float(score)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=max(1, args.trials), show_progress_bar=True)

    if study.best_trial is not None:
        best_score = float(study.best_value)
        best_params = dict(study.best_trial.params)
        best_params['long_link_cost_appearance_window_radius'] = 999999
        # Evaluate per-video once for reporting
        _, per_video = _evaluate_params_on_goldens(best_params, golden_paths, cache)
        all_results.append({'trial': 'best', 'params': best_params, 'score': best_score, 'per_video': per_video})

    if best_params is None:
        print('No comparable videos evaluated (missing goldens or key mismatches).')
        return

    print('\nBest params (avg pairwise_f1 = %.4f):' % best_score)
    for k in sorted(best_params.keys()):
        print(f'  {k}: {best_params[k]}')

    print('\nSuggested settings override (paste into settings or pass to DiscreteILPTracker):')
    print('{')
    for i, k in enumerate(sorted(best_params.keys())):
        v = best_params[k]
        comma = ',' if i < len(best_params) - 1 else ''
        print(
            f"    '{k}': {int(v) if isinstance(v, bool) or isinstance(v, int) else (f'{v:.3f}' if isinstance(v, float) else repr(v))}{comma}"
        )
    print('}')

    if args.report_json:
        out_path = Path(args.report_json)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'best_score': best_score, 'best_params': best_params, 'results': all_results}, f, indent=2)
        print(f'Wrote report to {out_path}')


if __name__ == '__main__':
    main()
