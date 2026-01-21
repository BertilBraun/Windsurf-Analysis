"""
Script to optimize Kalman Filter parameters using golden track annotations.
It minimizes the prediction error (1 - IoU) of the Kalman Filter against ground truth detections.

Usage:
    python server/inference/optimization/optimize_kalman.py --golden-dir tmp/golden --trials 100
"""

from __future__ import annotations

import sys
import optuna
import argparse
from pathlib import Path
from typing import Any, Dict, List

# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from video_processing.inference.src.motion.kalman_filter import _KalmanFilter, KFState
from video_processing.inference.src.motion.cmc import CMC
from video_processing.inference.src.common_types import BoundingBox
from video_processing.inference.src.visualization.stabilize import compute_stabilization_transforms_gmc
from video_processing.inference.optimization.optimization_util import optimize, list_golden_paths, load_full_tracks


def _evaluate_kalman(params: Dict[str, Any], data_cache: List[Dict[str, Any]]) -> float:
    total_iou_error = 0.0
    total_predictions = 0

    # Instantiate KF with params
    kf = _KalmanFilter(
        proc_std_weight_pos=float(params['proc_std_weight_pos']),
        proc_std_weight_vel=float(params['proc_std_weight_vel']),
        meas_std_weight_pos=float(params['meas_std_weight_pos']),
        meas_std_weight_size=float(params['meas_std_weight_size']),
        q_growth=float(params['q_growth']),
        dt=1.0,
    )

    for item in data_cache:
        tracks = item['tracks']
        transforms = item['transforms']
        cmc = CMC(transforms)

        for track in tracks:
            if len(track.sorted_detections) < 2:
                continue

            # Initialize state with first detection
            state = KFState.init(track.sorted_detections[0], kf=kf)

            # Iterate over subsequent detections
            for det in track.sorted_detections[1:]:
                # Predict to current detection frame
                pred_state = state.predict_to(det.frame_idx, cmc)

                # Extract predicted box (mean)
                # State mean is [cx, cy, w, h, ...]
                cx, cy, w, h = pred_state.mean[:4]
                # Ensure width and height are positive to avoid BoundingBox errors
                w = max(float(w), 1.0)
                h = max(float(h), 1.0)
                pred_bbox = BoundingBox.from_center_wh(cx, cy, w, h)

                # Compute error (1 - IoU)
                iou = pred_bbox.iou(det.bbox)
                total_iou_error += 1.0 - iou
                total_predictions += 1

                # Update state with true detection for next step
                state = state.update_to_det(det, cmc)

    if total_predictions == 0:
        return float('inf')

    # Return average 1-IoU
    return total_iou_error / total_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description='Optimize Kalman Filter parameters.')
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory with *.golden.tracks.pkl files')
    parser.add_argument('--trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Load all data and pre-compute transforms
    print('Loading data and computing transforms...')
    data_cache = []
    for path in list_golden_paths(args.golden_dir):
        tracks, meta = load_full_tracks(path)
        video_path = Path(meta.input_video_path)
        if not video_path.exists():
            print(f'Warning: Video file not found: {video_path}')
            continue

        try:
            transforms = compute_stabilization_transforms_gmc(video_path.as_posix())
            data_cache.append({'tracks': tracks, 'transforms': transforms})
        except Exception as e:
            print(f'Failed to compute transforms for {video_path}: {e}')

    print(f'Loaded {len(data_cache)} videos for optimization.')
    if not data_cache:
        print('No valid data found. Exiting.')
        return

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            # Logarithmic sampling for weights which can vary by orders of magnitude
            'proc_std_weight_pos': trial.suggest_float('proc_std_weight_pos', 1e-3, 0.2, log=True),
            'proc_std_weight_vel': trial.suggest_float('proc_std_weight_vel', 1e-4, 0.1, log=True),
            'meas_std_weight_pos': trial.suggest_float('meas_std_weight_pos', 1e-3, 0.5, log=True),
            'meas_std_weight_size': trial.suggest_float('meas_std_weight_size', 1e-3, 0.5, log=True),
            # q_growth is close to 1
            'q_growth': trial.suggest_float('q_growth', 1.0, 1.1),
        }

        return _evaluate_kalman(params, data_cache)

    # Minimize error (1 - IoU)
    optimize(objective, direction='minimize', trials=args.trials, seed=args.seed)


if __name__ == '__main__':
    main()
