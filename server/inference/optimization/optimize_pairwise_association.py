from __future__ import annotations

import sys
import optuna
import random
import argparse

from pathlib import Path
from typing import Callable, List, Tuple


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from server.inference.src.motion.cmc import CMC
from server.inference.src.visualization.stabilize import compute_stabilization_transforms_gmc
from server.inference.src.motion.kalman_filter import KFState
from server.inference.src.util.algebra import platt_prob_from_dist
from server.inference.optimization.optimization_util import each_golden, optimize
from server.inference.src.common_types import Track


# ----------------------------- Built-in configuration ----------------------------- #

# Global constants for optimization and sampling. Adjust here as needed.
MIN_SUB_LEN: int = 6
POS_PER_TRACK_CONTIG: int = 4
POS_PER_TRACK_NONCONTIG: int = 4


def _slice_track(track: Track, start_idx: int, end_idx_exclusive: int) -> Track:
    """Create a new Track from a contiguous slice of detections [start_idx, end_idx_exclusive)."""
    detections = track.sorted_detections[start_idx:end_idx_exclusive]

    if not detections:
        raise ValueError('Empty slice when creating sub-tracklet')

    return Track(track_id=track.track_id, sorted_detections=detections)


def _sample_positive_pairs(
    tracks: List[Track],
    *,
    min_sub_len: int,
    samples_per_track: int,
) -> List[Tuple[Track, Track]]:
    pairs: List[Tuple[Track, Track]] = []
    for t in tracks:
        n = len(t.sorted_detections)
        if n < 2 * min_sub_len:
            continue
        for _ in range(samples_per_track):
            split_lo = min_sub_len
            split_hi = n - min_sub_len
            k = random.randint(split_lo, split_hi)
            a = _slice_track(t, 0, k)
            b = _slice_track(t, k, n)
            pairs.append((a, b))
    return pairs


def _sample_positive_pairs_noncontiguous(
    tracks: List[Track],
    *,
    min_sub_len: int,
    samples_per_track: int,
) -> List[Tuple[Track, Track]]:
    pairs: List[Tuple[Track, Track]] = []
    for t in tracks:
        n = len(t.sorted_detections)
        if n < 2 * min_sub_len:
            continue
        indices = list(range(n))
        for _ in range(samples_per_track):
            # Draw two disjoint index sets of sizes >= min_sub_len
            if n < 2 * min_sub_len:
                break
            chosen = sorted(random.sample(indices, 2 * min_sub_len))
            idx_a = chosen[:min_sub_len]
            idx_b = chosen[min_sub_len:]
            a = Track(
                track_id=int(t.track_id),
                sorted_detections=[t.sorted_detections[i] for i in idx_a],
            )
            b = Track(
                track_id=int(t.track_id),
                sorted_detections=[t.sorted_detections[i] for i in idx_b],
            )
            pairs.append((a, b))
    return pairs


def _sample_negative_pairs(
    tracks: List[Track],
    *,
    samples_per_video: int,
    require_min_len: int,
) -> List[Tuple[Track, Track]]:
    # Use whole tracklets; filter by min length of detections
    eligible = [t for t in tracks if len(t.sorted_detections) >= require_min_len]
    pairs: List[Tuple[Track, Track]] = []
    if len(eligible) < 2:
        return pairs
    while len(pairs) < samples_per_video:
        a, b = random.sample(eligible, 2)
        if a.track_id == b.track_id:
            continue
        pairs.append((a, b))
    return pairs


def _build_dataset(golden_dir: str) -> Tuple[List[Track], List[Tuple[int, int, int, str]]]:
    # Build reusable dataset of (tracks, labeled pairs)
    dataset_tracks: List[Track] = []
    dataset_pairs: List[Tuple[int, int, int, str]] = []  # (i,j,label,video_path) with label 0=pos, 1=neg

    for tracks, meta in each_golden(golden_dir):
        video_path = meta.input_video_path

        pos_pairs: List[Tuple[Track, Track]] = []
        # always both positive modes
        pos_pairs.extend(
            _sample_positive_pairs(tracks, min_sub_len=MIN_SUB_LEN, samples_per_track=POS_PER_TRACK_CONTIG)
        )
        pos_pairs.extend(
            _sample_positive_pairs_noncontiguous(
                tracks, min_sub_len=MIN_SUB_LEN, samples_per_track=POS_PER_TRACK_NONCONTIG
            )
        )

        neg_target = len(pos_pairs)
        neg_pairs = _sample_negative_pairs(
            tracks, samples_per_video=neg_target, require_min_len=max(1, int(MIN_SUB_LEN))
        )
        # Collect unique tracklets to embed for this video
        unique_tracklets: List[Track] = []
        key_to_index: dict[int, int] = {}

        def _get_index(t: Track) -> int:
            k = id(t)
            idx = key_to_index.get(k)
            if idx is None:
                idx = len(unique_tracklets)
                unique_tracklets.append(t)
                key_to_index[k] = idx
            return idx

        pos_indices = [(_get_index(a), _get_index(b)) for (a, b) in pos_pairs]
        neg_indices = [(_get_index(a), _get_index(b)) for (a, b) in neg_pairs]

        base = len(dataset_tracks)
        dataset_tracks.extend(unique_tracklets)
        for i, j in pos_indices:
            dataset_pairs.append((base + i, base + j, 0, video_path))
        for i, j in neg_indices:
            dataset_pairs.append((base + i, base + j, 1, video_path))

    return dataset_tracks, dataset_pairs


def _balanced_mse(pos_costs: List[float], neg_costs: List[float]) -> float:
    if not pos_costs and not neg_costs:
        return float('nan')
    pos_mse = sum((c - 0.0) ** 2 for c in pos_costs) / max(1, len(pos_costs))
    neg_mse = sum((c - 1.0) ** 2 for c in neg_costs) / max(1, len(neg_costs))
    return 0.5 * pos_mse + 0.5 * neg_mse


def _run_optimization(
    get_params: Callable[[optuna.trial.Trial], dict],
    evaluate: Callable[[Track, Track, str, dict], float],
    args: argparse.Namespace,
) -> None:
    dataset_tracks, dataset_pairs = _build_dataset(args.golden_dir)

    def objective(trial: optuna.trial.Trial) -> float:
        params = get_params(trial)

        pos_costs: List[float] = []
        neg_costs: List[float] = []
        for i, j, label, video_path in dataset_pairs:
            c = evaluate(dataset_tracks[i], dataset_tracks[j], video_path, params)
            if label == 0:
                pos_costs.append(c)
            else:
                neg_costs.append(c)

        return _balanced_mse(pos_costs, neg_costs)

    optimize(objective, direction='minimize', trials=args.trials)


def optimize_embedding_distance(track_a: Track, track_b: Track, video_path: str, params: dict) -> float:
    """
    For mean embedding chi2 distance (objective = 0.099057)
    a: 7.427828328625088
    b: 4.088360175681194

    For mean embedding cosine similarity (objective = 0.253774):
    a: 0.0191976149872535
    b: -0.21355869883517353
    """

    gamma = float(params['gamma'])

    mean_embedding_a = track_a.mean_embedding()
    mean_embedding_b = track_b.mean_embedding()

    # d = mean_embedding_a.similarity(mean_embedding_b)
    # d = chi2_dist(mean_embedding_a, mean_embedding_b)
    # d = _calc_pairwise(track_a, track_b, lambda a, b: chi2_dist(a.embedding, b.embedding))  # too slow
    p_same = mean_embedding_a.probability(mean_embedding_b, gamma=gamma)
    return float(1.0 - p_same)  # 0 for positive, 1 for negative ideally


_motion_cache: dict[str, tuple[CMC, dict[int, KFState]]] = {}


def optimize_motion_distance(track_a: Track, track_b: Track, video_path: str, params: dict) -> float:
    """
    Best params (objective = 0.065495):
        a: 0.22599323488946937
        b: 4.992417880756955

    Best params (objective = 0.092563):
      a: 0.1626453944561744
      b: 3.153001966659187
    """
    if video_path not in _motion_cache:
        _motion_cache[video_path] = CMC(compute_stabilization_transforms_gmc(video_path)), {}
    cmc, track_end_kf_state = _motion_cache[video_path]
    if track_a.track_id not in track_end_kf_state:
        track_end_kf_state[track_a.track_id] = KFState.fit_kf_end_state(track_a.sorted_detections, cmc)
    if track_b.track_id not in track_end_kf_state:
        track_end_kf_state[track_b.track_id] = KFState.fit_kf_end_state(track_b.sorted_detections, cmc)

    a = float(params['a'])
    b = float(params['b'])

    # predict from A.end by Δ
    pred = track_end_kf_state[track_a.track_id].predict_to(track_b.start_frame, cmc)

    z_obs_back = track_b.sorted_detections[0].bbox.center_wh

    # position-only mahalanobis + log|S_pos|
    d2 = pred.gating_distance(z_obs_back, only_position=True)

    p_same = platt_prob_from_dist(d2, a, b)
    return float(1.0 - p_same)  # 0 for positive, 1 for negative ideally


def main() -> None:
    """This file exists to optimize the hyperparameters for the pairwise association cost function.
    It generates a dataset of Tracks which should match (Positives), and Tracks which should not match (Negatives).
    It then optimizes the hyperparameters for the pairwise association cost function which should return values close to 0 for Positives and close to 1 for Negatives.
    """

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory with *.golden.tracks.pkl files')
    parser.add_argument('--trials', type=int, default=1000, help='Number of Optuna trials')
    parser.add_argument(
        '--type', type=str, default='motion', choices=['motion', 'embedding'], help='Type of association to optimize'
    )

    args = parser.parse_args()

    if args.type == 'motion':
        _run_optimization(
            get_params=lambda trial: {
                'a': trial.suggest_float('a', 0.01, 10.0),
                'b': trial.suggest_float('b', -10.0, 10.0),
            },
            evaluate=optimize_motion_distance,
            args=args,
        )
    elif args.type == 'embedding':
        _run_optimization(
            get_params=lambda trial: {
                'gamma': trial.suggest_float('gamma', 0.01, 10.0),
            },
            evaluate=optimize_embedding_distance,
            args=args,
        )


if __name__ == '__main__':
    main()
