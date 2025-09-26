from __future__ import annotations

import sys
import math
import random
import argparse
import optuna

import numpy as np

from pathlib import Path
from typing import List, Tuple, Optional


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from server.inference.src.util.similarity_helpers import cosine_similarity, mean_embedding, _calc_pairwise, l2_normalize
from server.inference.src.player.core.player_state import Metadata, TrackLite, DetectionLite  # noqa: E402
from server.inference.src.common_types import BoundingBox, Detection, Track
from server.inference.src.util.video_io import VideoReader
from server.inference.src.tracking.reid import ReID, ReIDColorHistogram, ReIDOSNet, ReIDViT
from server.inference.src.settings import REID_MODEL_TYPE, OSNET_REID_MODEL_PATH


# ----------------------------- Built-in configuration ----------------------------- #

# Global constants for optimization and sampling. Adjust here as needed.
TRIALS: int = 80
RANDOM_SEED: int = 42
MIN_SUB_LEN: int = 3
POS_PER_TRACK_CONTIG: int = 1
POS_PER_TRACK_NONCONTIG: int = 1

# Define the search space for scorer parameters here.
# Type can be 'float' or 'int'; ranges are inclusive.
SEARCH_SPACE: dict = {
    'a': {'type': 'float', 'low': 0.01, 'high': 10.0},
    'b': {'type': 'float', 'low': -5.0, 'high': 5.0},
}


def _load_golden(path: Path) -> Metadata:
    import pickle

    with open(path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, Metadata):
        raise TypeError('Golden file does not contain Metadata')
    return data


def _init_reid_model(model_type: str) -> ReID:
    if model_type == 'color_hist':
        return ReIDColorHistogram()
    if model_type == 'osnet':
        return ReIDOSNet(model_path=OSNET_REID_MODEL_PATH)
    if model_type == 'vit':
        return ReIDViT()
    raise ValueError(f'Unknown REID_MODEL_TYPE: {model_type}')


def _clamp_bbox(b: List[int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _extract_embeddings_for_tracklets(video_path: str, tracklets: List[TrackLite]) -> List[List[np.ndarray]]:
    """Stream the video once and compute embeddings for each detection in the given tracklets.

    Returns a list parallel to `tracklets`, where each item is a list of embeddings aligned 1:1 with detections.
    """
    # Build frame -> items mapping
    frames_to_items: dict[int, List[Tuple[int, int, List[int]]]] = {}
    for ti, t in enumerate(tracklets):
        for di, det in enumerate(t.detections):
            frames_to_items.setdefault(int(det.frame_idx), []).append((ti, di, det.bbox))

    embeddings_by_tracklet: List[List[Optional[np.ndarray]]] = [[None for _ in t.detections] for t in tracklets]

    reid = _init_reid_model(REID_MODEL_TYPE)

    with VideoReader(video_path) as reader:
        props = reader.get_properties()
        width, height = int(props.width), int(props.height)
        for frame_idx, frame in reader.read_frames():
            pending = frames_to_items.get(int(frame_idx))
            if not pending:
                continue
            crops: List[np.ndarray] = []
            meta_indices: List[Tuple[int, int]] = []
            for ti, di, bbox in pending:
                x1, y1, x2, y2 = _clamp_bbox(bbox, width, height)
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0 or (y2 - y1) <= 0 or (x2 - x1) <= 0:
                    # Skip invalid crops; will fill zero vector later
                    meta_indices.append((ti, di))
                    crops.append(np.zeros((1, 1, 3), dtype=np.uint8))
                else:
                    meta_indices.append((ti, di))
                    crops.append(crop)

            feats = reid.get_features_for_crops(crops)
            assert feats.shape[0] == len(meta_indices)
            for (ti, di), feat in zip(meta_indices, feats):
                embeddings_by_tracklet[ti][di] = feat.astype(np.float32, copy=False)

    # Fill any missing with zeros and cast to plain lists
    feature_dim: Optional[int] = None
    for per_track in embeddings_by_tracklet:
        for e in per_track:
            if e is not None:
                feature_dim = int(e.shape[0])
                break
        if feature_dim is not None:
            break

    finalized: List[List[np.ndarray]] = []
    for per_track in embeddings_by_tracklet:
        out_track: List[np.ndarray] = []
        for e in per_track:
            if e is None:
                dim = feature_dim if feature_dim is not None else 128
                out_track.append(np.zeros((dim,), dtype=np.float32))
            else:
                out_track.append(e)
        finalized.append(out_track)
    return finalized


def _to_track_with_embeddings(tl: TrackLite, embeddings: List[np.ndarray]) -> Track:
    dets: List[Detection] = []
    for det_lite, emb in zip(tl.detections, embeddings):
        x1, y1, x2, y2 = det_lite.bbox
        dets.append(
            Detection(
                bbox=BoundingBox(int(x1), int(y1), int(x2), int(y2)),
                embedding=np.asarray(emb, dtype=np.float32),
                confidence=float(det_lite.confidence),
                frame_idx=int(det_lite.frame_idx),
            )
        )
    return Track(track_id=int(tl.track_id), sorted_detections=dets)


def _slice_tracklite(track: TrackLite, start_idx: int, end_idx_exclusive: int, fps: float) -> TrackLite:
    """Create a new TrackLite from a contiguous slice of detections [start_idx, end_idx_exclusive)."""
    dets = track.detections[start_idx:end_idx_exclusive]
    if not dets:
        raise ValueError('Empty slice when creating sub-tracklet')

    start_frame = int(dets[0].frame_idx)
    end_frame = int(dets[-1].frame_idx)
    duration_frames = max(0, end_frame - start_frame + 1)

    return TrackLite(
        track_id=int(track.track_id),
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=float(start_frame / max(1.0, fps)),
        duration=float(duration_frames / max(1.0, fps)),
        detection_count=len(dets),
        detections=[
            DetectionLite(frame_idx=int(d.frame_idx), bbox=[int(b) for b in d.bbox], confidence=float(d.confidence))
            for d in dets
        ],
    )


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float('nan'), float('nan')
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float('nan')
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def _sample_positive_pairs_noncontiguous(
    tracks: List[TrackLite],
    *,
    min_sub_len: int,
    samples_per_track: int,
    rng: random.Random,
    fps: float,
) -> List[Tuple[TrackLite, TrackLite]]:
    pairs: List[Tuple[TrackLite, TrackLite]] = []
    for t in tracks:
        n = int(t.detection_count)
        if n < 2 * min_sub_len:
            continue
        indices = list(range(n))
        for _ in range(samples_per_track):
            # Draw two disjoint index sets of sizes >= min_sub_len
            if n < 2 * min_sub_len:
                break
            chosen = rng.sample(indices, 2 * min_sub_len)
            rng.shuffle(chosen)
            idx_a = sorted(chosen[:min_sub_len])
            idx_b = sorted(chosen[min_sub_len:])
            a = TrackLite(
                track_id=int(t.track_id),
                start_frame=int(t.detections[idx_a[0]].frame_idx),
                end_frame=int(t.detections[idx_a[-1]].frame_idx),
                start_time=float(t.detections[idx_a[0]].frame_idx / max(1.0, fps)),
                duration=float(
                    (t.detections[idx_a[-1]].frame_idx - t.detections[idx_a[0]].frame_idx + 1) / max(1.0, fps)
                ),
                detection_count=len(idx_a),
                detections=[t.detections[i] for i in idx_a],
            )
            b = TrackLite(
                track_id=int(t.track_id),
                start_frame=int(t.detections[idx_b[0]].frame_idx),
                end_frame=int(t.detections[idx_b[-1]].frame_idx),
                start_time=float(t.detections[idx_b[0]].frame_idx / max(1.0, fps)),
                duration=float(
                    (t.detections[idx_b[-1]].frame_idx - t.detections[idx_b[0]].frame_idx + 1) / max(1.0, fps)
                ),
                detection_count=len(idx_b),
                detections=[t.detections[i] for i in idx_b],
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
    tracks: List[TrackLite],
    fps: float,
    *,
    min_sub_len: int,
    samples_per_track: int,
    rng: random.Random,
) -> List[Tuple[TrackLite, TrackLite]]:
    pairs: List[Tuple[TrackLite, TrackLite]] = []
    for t in tracks:
        n = int(t.detection_count)
        if n < 2 * min_sub_len:
            continue
        for _ in range(samples_per_track):
            split_lo = min_sub_len
            split_hi = n - min_sub_len
            if split_lo >= split_hi:
                break
            k = rng.randint(split_lo, split_hi)
            a = _slice_tracklite(t, 0, k, fps)
            b = _slice_tracklite(t, k, n, fps)
            pairs.append((a, b))
    return pairs


def _sample_negative_pairs(
    tracks: List[TrackLite],
    *,
    samples_per_video: int,
    rng: random.Random,
    require_min_len: int,
) -> List[Tuple[TrackLite, TrackLite]]:
    # Use whole tracklets; filter by min length of detections
    eligible = [t for t in tracks if int(t.detection_count) >= require_min_len]
    pairs: List[Tuple[TrackLite, TrackLite]] = []
    if len(eligible) < 2:
        return pairs
    for _ in range(samples_per_video):
        a, b = rng.sample(eligible, 2)
        if int(a.track_id) == int(b.track_id):
            continue
        pairs.append((a, b))
    return pairs


def _builtin_cost(track_a: Track, track_b: Track, params: dict) -> float:
    def chi2_dist(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
        """Calculate the chi2 distance between two embeddings."""
        num = (p - q) ** 2
        den = p + q + eps
        return 0.5 * float((num / den).sum())

    def platt_prob_from_dist(d: float, a: float, b: float) -> float:
        """Calculate the probability for a distance to say, that the two tracks are the same. `a` and `b` are parameters of the platt scaling. The returned probability is in the range [0, 1] (sigmoid(a * -d + b))"""
        z = a * (-d) + b
        p = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(p, 1e-6, 1 - 1e-6))

    a = float(params.get('a', 1.0))
    b = float(params.get('b', 0.0))

    mean_embedding_a = mean_embedding(track_a)
    mean_embedding_b = mean_embedding(track_b)

    d = cosine_similarity(mean_embedding_a, mean_embedding_b)
    d = chi2_dist(mean_embedding_a, mean_embedding_b)
    # d = _calc_pairwise(track_a, track_b, lambda a, b: chi2_dist(a.embedding, b.embedding))  # too slow
    p_same = platt_prob_from_dist(d, a, b)
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

    rng = random.Random(RANDOM_SEED)

    def score_with_params(a: Track, b: Track, video_path: str, params: dict) -> float:
        return _builtin_cost(a, b, params)

    golden_paths = sorted(Path(args.golden_dir).glob('*.golden.tracks.pkl'))
    if not golden_paths:
        print('No golden files found in the specified directory.')
        return

    # Build reusable dataset of (tracks, labeled pairs)
    dataset_tracks: List[Track] = []
    dataset_pairs: List[Tuple[int, int, int, str]] = []  # (i,j,label,video_path) with label 0=pos, 1=neg
    per_video_stats = []

    for gpath in golden_paths:
        meta = _load_golden(gpath)
        video_path = meta.input_video_path
        fps = float(getattr(meta.video_properties, 'fps', 30.0))
        tracks = list(meta.tracks)

        pos_pairs: List[Tuple[TrackLite, TrackLite]] = []
        # always both positive modes
        pos_pairs.extend(
            _sample_positive_pairs(
                tracks, fps, min_sub_len=MIN_SUB_LEN, samples_per_track=POS_PER_TRACK_CONTIG, rng=rng
            )
        )
        pos_pairs.extend(
            _sample_positive_pairs_noncontiguous(
                tracks, min_sub_len=MIN_SUB_LEN, samples_per_track=POS_PER_TRACK_NONCONTIG, rng=rng, fps=fps
            )
        )

        neg_target = len(pos_pairs)
        neg_pairs = _sample_negative_pairs(
            tracks, samples_per_video=neg_target, rng=rng, require_min_len=max(1, int(MIN_SUB_LEN))
        )
        # Collect unique tracklets to embed for this video
        unique_tracklets: List[TrackLite] = []
        key_to_index: dict[int, int] = {}

        def _get_index(t: TrackLite) -> int:
            k = id(t)
            idx = key_to_index.get(k)
            if idx is None:
                idx = len(unique_tracklets)
                unique_tracklets.append(t)
                key_to_index[k] = idx
            return idx

        pos_indices = [(_get_index(a), _get_index(b)) for (a, b) in pos_pairs]
        neg_indices = [(_get_index(a), _get_index(b)) for (a, b) in neg_pairs]

        embeddings_list = _extract_embeddings_for_tracklets(video_path, unique_tracklets)

        # Convert to full Tracks with embeddings
        full_tracks: List[Track] = [
            _to_track_with_embeddings(tl, emb) for tl, emb in zip(unique_tracklets, embeddings_list)
        ]

        base = len(dataset_tracks)
        dataset_tracks.extend(full_tracks)
        for i, j in pos_indices:
            dataset_pairs.append((base + i, base + j, 0, video_path))
        for i, j in neg_indices:
            dataset_pairs.append((base + i, base + j, 1, video_path))

        per_video_stats.append(
            {
                'video': Path(video_path).name,
                'golden_file': gpath.name,
                'num_tracks': len(tracks),
                'pos_pairs': len(pos_pairs),
                'neg_pairs': len(neg_pairs),
            }
        )

    # Always optimize using built-in SEARCH_SPACE
    def suggest(trial: optuna.trial.Trial) -> dict:
        params: dict = {}
        for name, spec in SEARCH_SPACE.items():
            t = spec.get('type', 'float')
            if t == 'int':
                params[name] = trial.suggest_int(name, int(spec['low']), int(spec['high']))
            elif t == 'float':
                params[name] = trial.suggest_float(name, float(spec['low']), float(spec['high']))
            else:
                raise ValueError(f'Unsupported param type: {t}')
        return params

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest(trial)
        pos_costs: List[float] = []
        neg_costs: List[float] = []
        for i, j, label, vp in dataset_pairs:
            c = score_with_params(dataset_tracks[i], dataset_tracks[j], vp, params)
            if label == 0:
                pos_costs.append(c)
            else:
                neg_costs.append(c)
        return _balanced_mse(pos_costs, neg_costs)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=max(1, TRIALS), show_progress_bar=True)

    print('\nBest params (objective = %.6f):' % float(study.best_value))
    best = dict(study.best_trial.params)
    for k in sorted(best.keys()):
        print(f'  {k}: {best[k]}')


if __name__ == '__main__':
    main()
