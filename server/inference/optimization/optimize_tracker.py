from __future__ import annotations

import sys
import json
import math
import argparse
from dataclasses import dataclass
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
from server.inference.src.tracking.bot_sort import BotSortTracker
from server.inference.src.common_types import Detection, Track
from server.inference.src.player.core.player_state import Metadata, TrackLite, DetectionLite
from server.inference.src.settings import YOLO_MODEL_PATH


def _detections_to_initial_tracks(detections: List[Detection]) -> List[Track]:
    return [Track(track_id=i + 1, sorted_detections=[det]) for i, det in enumerate(detections)]


def _load_golden(path: Path) -> Metadata:
    import pickle

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
        'pairwise_precision': float(prec),
        'pairwise_recall': float(rec),
        'pairwise_f1': float(f1),
        'rand_index': float(rand),
        'jaccard_same': float(jaccard_same),
    }


# ───────────────────────────── Preprocessor mode ───────────────────────────── #


def _evaluate_preprocessor(
    params: Dict[str, Any], golden_paths: List[Path], detector: SurferDetector
) -> tuple[float, List[Dict[str, Any]]]:
    logs: List[Dict[str, Any]] = []
    total_tracks = 0
    num_videos = 0
    total_impure = 0

    for gpath in golden_paths:
        meta = _load_golden(gpath)
        video_path = Path(meta.input_video_path)

        props = get_video_properties(video_path)
        dets = detector.run_object_detection_on_video(video_path.as_posix())

        pre = Preprocessor(
            greedy_min_iou=float(params['greedy_min_iou']),
            greedy_min_cosine_similarity=float(params['greedy_min_cos']),
            greedy_max_frame_distance=int(params['greedy_max_gap']),
            greedy_ema_alpha=float(params['greedy_ema_alpha']),
        )
        initial_tracks = _detections_to_initial_tracks(dets)
        pred_tracks = pre.track(list(initial_tracks), props)

        gold_assign = _build_assignment_from_metadata(meta)
        pred_assign = _build_assignment_from_tracks(pred_tracks)

        if set(gold_assign.keys()) != set(pred_assign.keys()):
            logs.append({'video': gpath.name, 'tracks': len(pred_tracks), 'impure_tracks': 'key_mismatch'})
            total_impure += 1
            num_videos += 1
            total_tracks += len(pred_tracks)
            continue

        from collections import defaultdict

        gold_ids_by_pred: Dict[int, set[int]] = defaultdict(set)
        for det_key, pred_tid in pred_assign.items():
            gold_tid = gold_assign[det_key]
            gold_ids_by_pred[int(pred_tid)].add(int(gold_tid))

        impure = sum(1 for s in gold_ids_by_pred.values() if len(s) > 1)
        total_impure += impure
        total_tracks += len(gold_ids_by_pred)
        num_videos += 1

        logs.append({'video': gpath.name, 'tracks': len(gold_ids_by_pred), 'impure_tracks': impure})

    if num_videos == 0:
        return -1e9, logs

    avg_tracks = total_tracks / max(1, num_videos)
    if total_impure > 0:
        score = -1000.0 * float(total_impure) - avg_tracks
    else:
        score = 1000.0 - avg_tracks
    return float(score), logs


def _run_opt_preprocessor(args) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    golden_dir = Path(args.golden_dir)
    golden_paths = sorted(golden_dir.glob('*.golden.tracks.pkl'))
    if not golden_paths:
        print('No golden files found in the specified directory.')
        return -math.inf, {}, []

    detector = SurferDetector(yolo_model_path=YOLO_MODEL_PATH)

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'greedy_min_iou': trial.suggest_float('greedy_min_iou', 0.0, 0.7),
            'greedy_min_cos': trial.suggest_float('greedy_min_cos', 0.5, 0.99),
            'greedy_max_gap': trial.suggest_int('greedy_max_gap', 0, 30),
            'greedy_ema_alpha': trial.suggest_float('greedy_ema_alpha', 0.0, 1.0),
        }
        score, _ = _evaluate_preprocessor(params, golden_paths, detector)
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=max(1, args.trials), show_progress_bar=True)

    if study.best_trial is None:
        print('Optimization produced no result.')
        return -math.inf, {}, []

    best_params = dict(study.best_trial.params)
    best_score, logs = _evaluate_preprocessor(best_params, golden_paths, detector)
    return float(best_score), dict(best_params), logs


# ───────────────────────────── Discrete ILP mode ───────────────────────────── #


class _DetectionCache:
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
            dets = self._detector.run_object_detection_on_video(video_path.as_posix())
            self._detections_by_video[video_path] = dets

        initial_tracks = _detections_to_initial_tracks(dets)
        tracks = Preprocessor().track(initial_tracks, video_props)
        self._preprocessed_tracks_by_video[video_path] = (tracks, video_props)
        return tracks, video_props


def _evaluate_discrete(
    params: Dict[str, Any], golden_paths: List[Path], cache: _DetectionCache
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
            print(f'Detections differ for {golden_path.name}')
            continue
        s = _pairwise_scores(gold_assign, pred_assign)
        metrics_list.append((golden_path.name, s))

    if not metrics_list:
        return float('nan'), []

    avg_f1 = sum(s['pairwise_f1'] for _, s in metrics_list) / len(metrics_list)
    return float(avg_f1), metrics_list


def _run_opt_discrete(args) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    golden_dir = Path(args.golden_dir)
    if not golden_dir.exists():
        print(f'Golden dir not found: {golden_dir}')
        return -math.inf, {}, []

    golden_paths = sorted(golden_dir.glob('*.golden.tracks.pkl'))
    if not golden_paths:
        print('No golden files found in the specified directory.')
        return -math.inf, {}, []

    cache = _DetectionCache()

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
        score, _ = _evaluate_discrete(params, golden_paths, cache)
        return -1.0 if math.isnan(score) else float(score)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=max(1, args.trials), show_progress_bar=True)

    if study.best_trial is None:
        print('No comparable videos evaluated (missing goldens or key mismatches).')
        return -math.inf, {}, []

    best_score = float(study.best_value)
    best_params = dict(study.best_trial.params)
    best_params['long_link_cost_appearance_window_radius'] = 999999
    _, per_video = _evaluate_discrete(best_params, golden_paths, cache)
    per_vid_logs = [{'video': v, **m} for v, m in per_video]
    return float(best_score), dict(best_params), per_vid_logs


# ─────────────────────────────── BoT-SORT mode ─────────────────────────────── #


@dataclass
class _IoUMatch:
    iou: float
    gold_key: Tuple[int, int, int, int, int]


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _evaluate_botsort(
    params: Dict[str, Any],
    golden_paths: List[Path],
    detector: SurferDetector,
    iou_map_thresh: float = 0.5,
) -> Tuple[float, List[Tuple[str, Dict[str, float]]]]:
    metrics_list: List[Tuple[str, Dict[str, float]]] = []

    for gpath in golden_paths:
        gold_meta = _load_golden(gpath)
        video_path = Path(gold_meta.input_video_path)

        props = get_video_properties(video_path)
        dets = detector.run_object_detection_on_video(video_path.as_posix())

        # Prepare input for BoT-SORT (one detection per initial Track)
        initial_tracks = _detections_to_initial_tracks(dets)
        tracker = BotSortTracker(
            vid_file_path=video_path.as_posix(),
            track_high_thresh=float(params['track_high_thresh']),
            track_low_thresh=float(params['track_low_thresh']),
            new_track_thresh=float(params['new_track_thresh']),
            track_buffer=int(params['track_buffer']),
            proximity_thresh=float(params['proximity_thresh']),
            appearance_thresh=float(params['appearance_thresh']),
            match_thresh=float(params['match_thresh']),
            cmc_method='sparseOptFlow',
        )

        pred_tracks = tracker.track(initial_tracks, props, transforms=[])

        # Build gold and raw-detection key sets
        gold_assign = _build_assignment_from_metadata(gold_meta)

        # Index raw detections by frame for matching
        dets_by_frame: Dict[int, List[Detection]] = {}
        for d in dets:
            dets_by_frame.setdefault(int(d.frame_idx), []).append(d)

        # Map predicted detections to raw detection keys via IoU on the same frame
        pred_assign: Dict[Tuple[int, int, int, int, int], int] = {}
        best_iou_by_key: Dict[Tuple[int, int, int, int, int], float] = {}
        for t in pred_tracks:
            for det in t.sorted_detections:
                f = int(det.frame_idx)
                raw_list = dets_by_frame.get(f, [])
                if not raw_list:
                    continue
                best: Optional[_IoUMatch] = None
                pb = (int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2))
                for rd in raw_list:
                    rb = (int(rd.bbox.x1), int(rd.bbox.y1), int(rd.bbox.x2), int(rd.bbox.y2))
                    iou = _bbox_iou(pb, rb)
                    if iou >= iou_map_thresh:
                        key = (f, rb[0], rb[1], rb[2], rb[3])
                        if best is None or iou > best.iou:
                            best = _IoUMatch(iou=iou, gold_key=key)
                if best is not None:
                    # If multiple preds map to same raw key, keep the higher IoU one
                    prev_best = best_iou_by_key.get(best.gold_key, -1.0)
                    if best.iou > prev_best:
                        pred_assign[best.gold_key] = int(t.track_id)
                        best_iou_by_key[best.gold_key] = best.iou

        # Fill in any unmatched gold detections with unique dummy track ids
        if set(gold_assign.keys()) != set(pred_assign.keys()):
            missing = [k for k in gold_assign.keys() if k not in pred_assign]
            if missing:
                # Offset dummy ids past current max predicted id
                next_tid = (max(pred_assign.values()) + 1) if pred_assign else 1
                for k in missing:
                    pred_assign[k] = next_tid
                    next_tid += 1

        s = _pairwise_scores(gold_assign, pred_assign)
        metrics_list.append((gpath.name, s))

    if not metrics_list:
        return float('nan'), []

    avg_f1 = sum(s['pairwise_f1'] for _, s in metrics_list) / len(metrics_list)
    return float(avg_f1), metrics_list


def _run_opt_botsort(args) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    golden_dir = Path(args.golden_dir)
    golden_paths = sorted(golden_dir.glob('*.golden.tracks.pkl'))
    if not golden_paths:
        print('No golden files found in the specified directory.')
        return -math.inf, {}, []

    detector = SurferDetector(yolo_model_path=YOLO_MODEL_PATH)

    def objective(trial: optuna.trial.Trial) -> float:
        params: Dict[str, Any] = {
            'track_high_thresh': trial.suggest_float('track_high_thresh', 0.3, 0.9),
            'track_low_thresh': trial.suggest_float('track_low_thresh', 0.01, 0.3),
            'new_track_thresh': trial.suggest_float('new_track_thresh', 0.4, 0.9),
            'track_buffer': trial.suggest_int('track_buffer', 10, 150),
            'proximity_thresh': trial.suggest_float('proximity_thresh', 0.3, 0.9),
            'appearance_thresh': trial.suggest_float('appearance_thresh', 0.3, 0.9),
            'match_thresh': trial.suggest_float('match_thresh', 0.5, 0.95),
        }
        score, _ = _evaluate_botsort(params, golden_paths, detector, iou_map_thresh=float(args.iou_map_thresh))
        return -1.0 if math.isnan(score) else float(score)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=max(1, args.trials), show_progress_bar=True)

    if study.best_trial is None:
        print('Optimization produced no result.')
        return -math.inf, {}, []

    best_params = dict(study.best_trial.params)
    best_score, logs = _evaluate_botsort(best_params, golden_paths, detector, iou_map_thresh=float(args.iou_map_thresh))
    per_vid_logs = [{'video': v, **m} for v, m in logs]
    return float(best_score), dict(best_params), per_vid_logs


# ─────────────────────────────────── CLI ───────────────────────────────────── #


def main() -> None:
    parser = argparse.ArgumentParser(description='Unified optimizer for Preprocessor, Discrete ILP, and BoT-SORT.')
    parser.add_argument('--mode', type=str, required=True, choices=['preprocessor', 'ilp', 'botsort'])
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory with *.golden.tracks.pkl files')
    parser.add_argument('--trials', type=int, default=80, help='Number of Optuna trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--report-json', type=str, default='', help='Optional path to write detailed results JSON')
    parser.add_argument('--iou-map-thresh', type=float, default=0.5, help='IoU threshold for BoT-SORT mapping')

    args = parser.parse_args()

    mode = args.mode
    best_score: float
    best_params: Dict[str, Any]
    logs: List[Dict[str, Any]]

    if mode == 'preprocessor':
        best_score, best_params, logs = _run_opt_preprocessor(args)
    elif mode == 'ilp':
        best_score, best_params, logs = _run_opt_discrete(args)
    elif mode == 'botsort':
        best_score, best_params, logs = _run_opt_botsort(args)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    if not best_params:
        return

    print('\nBest params (score = %.4f):' % float(best_score))
    for k in sorted(best_params.keys()):
        print(f'  {k}: {best_params[k]}')

    if mode == 'ilp':
        print('\nSuggested settings override (paste into settings or pass to DiscreteILPTracker):')
        print('{')
        keys = sorted(best_params.keys())
        for i, k in enumerate(keys):
            v = best_params[k]
            comma = ',' if i < len(keys) - 1 else ''
            print(
                f"    '{k}': {int(v) if isinstance(v, bool) or isinstance(v, int) else (f'{v:.3f}' if isinstance(v, float) else repr(v))}{comma}"
            )
        print('}')

    if args.report_json:
        out_path = Path(args.report_json)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'mode': mode, 'best_score': best_score, 'best_params': best_params, 'logs': logs}, f, indent=2)
        print(f'Wrote report to {out_path}')


if __name__ == '__main__':
    main()
