from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import optuna


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from server.inference.src.util.video_io import get_video_properties
from server.inference.src.tracking.detector import SurferDetector
from server.inference.src.tracking.preprocessing.preprocessor import Preprocessor
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


def _evaluate_params(
    params: Dict[str, Any],
    golden_paths: List[Path],
    detector: SurferDetector,
) -> tuple[float, List[Dict[str, Any]]]:
    """Return (score, per_video_logs). Score favors purity (no mixed gold IDs) and fewer tracks.

    - If any predicted track contains detections from multiple gold tracks, apply a strong penalty.
    - Otherwise, aim to minimize the average number of tracks across videos.
    """
    logs: List[Dict[str, Any]] = []
    total_tracks = 0
    num_videos = 0
    total_impure = 0

    for gpath in golden_paths:
        meta = _load_golden(gpath)
        video_path = Path(meta.input_video_path)

        props = get_video_properties(video_path)
        dets = detector.run_object_detection_on_video(video_path.as_posix())

        # Preprocess only (no tracker) with proposed params
        pre = Preprocessor(
            greedy_min_iou=float(params['greedy_min_iou']),
            greedy_min_cosine_similarity=float(params['greedy_min_cos']),
            greedy_max_frame_distance=int(params['greedy_max_gap']),
            greedy_ema_alpha=float(params['greedy_ema_alpha']),
        )
        initial_tracks = _detections_to_initial_tracks(dets)
        pred_tracks = pre.track(list(initial_tracks), props)

        # Compare to gold
        gold_assign = _build_assignment_from_metadata(meta)
        pred_assign = _build_assignment_from_tracks(pred_tracks)

        # Keys must match (same detections); otherwise treat as impure
        if set(gold_assign.keys()) != set(pred_assign.keys()):
            logs.append({'video': gpath.name, 'tracks': len(pred_tracks), 'impure_tracks': 'key_mismatch'})
            total_impure += 1
            num_videos += 1
            total_tracks += len(pred_tracks)
            continue

        # For each predicted track, check purity: all detections share one gold id
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
        # Strong penalty for any impurity
        score = -1000.0 * float(total_impure) - avg_tracks
    else:
        # Pure → reward fewer tracks
        score = 1000.0 - avg_tracks
    return float(score), logs


def main() -> None:
    parser = argparse.ArgumentParser(description='Bayesian optimization of Preprocessor for pure minimal tracklets.')
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory with *.golden.tracks.pkl files')
    parser.add_argument('--trials', type=int, default=80, help='Number of Optuna trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--report-json', type=str, default='', help='Optional path to write detailed results JSON')
    args = parser.parse_args()

    golden_dir = Path(args.golden_dir)
    golden_paths = sorted(golden_dir.glob('*.golden.tracks.pkl'))
    if not golden_paths:
        print('No golden files found in the specified directory.')
        return

    detector = SurferDetector(yolo_model_path=YOLO_MODEL_PATH)

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'greedy_min_iou': trial.suggest_float('greedy_min_iou', 0.0, 0.7),
            'greedy_min_cos': trial.suggest_float('greedy_min_cos', 0.5, 0.99),
            'greedy_max_gap': trial.suggest_int('greedy_max_gap', 0, 30),
            'greedy_ema_alpha': trial.suggest_float('greedy_ema_alpha', 0.0, 1.0),
        }
        score, _ = _evaluate_params(params, golden_paths, detector)
        return float(score)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=max(1, args.trials), show_progress_bar=True)

    if study.best_trial is None:
        print('Optimization produced no result.')
        return

    best_params = dict(study.best_trial.params)
    best_score, logs = _evaluate_params(best_params, golden_paths, detector)

    print('Best preprocessor params:')
    for k in sorted(best_params.keys()):
        print(f'  {k}: {best_params[k]}')
    print(f'Aggregate score: {best_score:.4f}')

    print('Per-video:')
    for l in logs:
        print(f'  {l["video"]}: tracks={l["tracks"]} impure={l["impure_tracks"]}')

    if args.report_json:
        out = {
            'best_score': best_score,
            'best_params': best_params,
            'per_video': logs,
        }
        with open(args.report_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f'Wrote report to {args.report_json}')


if __name__ == '__main__':
    main()
