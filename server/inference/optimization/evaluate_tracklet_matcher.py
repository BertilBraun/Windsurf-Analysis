from __future__ import annotations

import sys
import random
import argparse
import optuna

import numpy as np

from pathlib import Path
from typing import List, Tuple

from server.inference.src.util.similarity_helpers import HistogramEmbedding


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from server.inference.optimization.optimization_util import each_golden, optimize
from server.inference.src.common_types import Track


# ----------------------------- Built-in configuration ----------------------------- #

# Global constants for optimization and sampling. Adjust here as needed.
MIN_SUB_LEN: int = 3
POS_PER_TRACK_CONTIG: int = 1
POS_PER_TRACK_NONCONTIG: int = 1


def _slice_track(track: Track, start_idx: int, end_idx_exclusive: int) -> Track:
    """Create a new Track from a contiguous slice of detections [start_idx, end_idx_exclusive)."""
    detections = track.sorted_detections[start_idx:end_idx_exclusive]

    if not detections:
        raise ValueError('Empty slice when creating sub-tracklet')

    return Track(track_id=track.track_id, sorted_detections=detections)


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
            chosen = random.sample(indices, 2 * min_sub_len)
            random.shuffle(chosen)
            idx_a = sorted(chosen[:min_sub_len])
            idx_b = sorted(chosen[min_sub_len:])
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


def _balanced_mse(pos_costs: List[float], neg_costs: List[float]) -> float:
    if not pos_costs and not neg_costs:
        return float('nan')
    pos_mse = sum((c - 0.0) ** 2 for c in pos_costs) / max(1, len(pos_costs))
    neg_mse = sum((c - 1.0) ** 2 for c in neg_costs) / max(1, len(neg_costs))
    return 0.5 * pos_mse + 0.5 * neg_mse


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
            if split_lo >= split_hi:
                break
            k = random.randint(split_lo, split_hi)
            a = _slice_track(t, 0, k)
            b = _slice_track(t, k, n)
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
    for _ in range(samples_per_video):
        a, b = random.sample(eligible, 2)
        if int(a.track_id) == int(b.track_id):
            continue
        pairs.append((a, b))
    return pairs


def _builtin_cost(track_a: Track, track_b: Track, params: dict) -> float:
    def platt_prob_from_dist(d: float, a: float, b: float) -> float:
        """Calculate the probability for a distance to say, that the two tracks are the same. `a` and `b` are parameters of the platt scaling. The returned probability is in the range [0, 1] (sigmoid(a * -d + b))"""
        z = a * (-d) + b
        p = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(p, 1e-6, 1 - 1e-6))

    a = float(params.get('a', 1.0))
    b = float(params.get('b', 0.0))

    mean_embedding_a = track_a.mean_embedding()
    mean_embedding_b = track_b.mean_embedding()

    # d = mean_embedding_a.similarity(mean_embedding_b)
    # d = chi2_dist(mean_embedding_a, mean_embedding_b)
    # d = _calc_pairwise(track_a, track_b, lambda a, b: chi2_dist(a.embedding, b.embedding))  # too slow
    assert isinstance(mean_embedding_a, HistogramEmbedding)
    assert isinstance(mean_embedding_b, HistogramEmbedding)
    p_same = mean_embedding_a.probability(mean_embedding_b, a, b)
    return float(1.0 - p_same)  # 0 for positive, 1 for negative ideally


"""  
For mean embedding chi2 distance (objective = 0.099057)
a: 7.427828328625088
b: 4.088360175681194

For mean embedding cosine similarity (objective = 0.253774):
a: 0.0191976149872535
b: -0.21355869883517353

"""


def main() -> None:
    parser = argparse.ArgumentParser(description='Optimize built-in tracklet matching scorer on golden tracklets.')
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory with *.golden.tracks.pkl files')
    # no external config; everything driven by constants above

    args = parser.parse_args()

    def score_with_params(a: Track, b: Track, video_path: str, params: dict) -> float:
        return _builtin_cost(a, b, params)

    # Build reusable dataset of (tracks, labeled pairs)
    dataset_tracks: List[Track] = []
    dataset_pairs: List[Tuple[int, int, int, str]] = []  # (i,j,label,video_path) with label 0=pos, 1=neg
    per_video_stats = []

    for tracks, meta in each_golden(args.golden_dir):
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
        dataset_tracks.extend(tracks)
        for i, j in pos_indices:
            dataset_pairs.append((base + i, base + j, 0, video_path))
        for i, j in neg_indices:
            dataset_pairs.append((base + i, base + j, 1, video_path))

        per_video_stats.append(
            {
                'video': Path(video_path).name,
                'num_tracks': len(tracks),
                'pos_pairs': len(pos_pairs),
                'neg_pairs': len(neg_pairs),
            }
        )

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'a': trial.suggest_float('a', 0.01, 10.0),
            'b': trial.suggest_float('b', -5.0, 5.0),
        }

        pos_costs: List[float] = []
        neg_costs: List[float] = []
        for i, j, label, vp in dataset_pairs:
            c = score_with_params(dataset_tracks[i], dataset_tracks[j], vp, params)
            if label == 0:
                pos_costs.append(c)
            else:
                neg_costs.append(c)

        return _balanced_mse(pos_costs, neg_costs)

    optimize(objective, direction='minimize')


if __name__ == '__main__':
    main()
