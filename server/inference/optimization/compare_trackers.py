from __future__ import annotations

import sys
import argparse
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from server.inference.src.tracking.tracking import Tracker
from server.inference.src.util.similarity_helpers import l2_normalize
from server.inference.src.player.core.player_state import Metadata, TrackLite, DetectionLite
from server.inference.src.common_types import BoundingBox, Detection, Track
from server.inference.src.util.video_io import get_video_properties, VideoInfo
from server.inference.src.tracking.reid import ReID, ReIDColorHistogram, ReIDOSNet, ReIDViT
from server.inference.src.settings import REID_MODEL_TYPE, OSNET_REID_MODEL_PATH
from server.inference.src.tracking.bot_sort import BotSortTracker
from server.inference.src.tracking.discrete_opt_tracker import DiscreteOptTracker
from server.inference.src.tracking.iterative_ilp_tracker import IterativeILPTracker
from server.inference.src.tracking.oc_sort import OCSortEmbedTracker
from server.inference.src.tracking.preprocessing.preprocessor import Preprocessor
from server.inference.src.tracking.track_processing import (
    TrackFiltering,
    TrackInterpolation,
    TrackSmoothing,
    TrackRelabeling,
)
from server.inference.src.visualization.stabilize import compute_stabilization_transforms_gmc, Transform
from server.inference.src.util.video_io import VideoReader


# ─────────────────────────────── helpers ─────────────────────────────── #


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


def _extract_embeddings_for_gold(video_path: str, gold_tracks: List[TrackLite], reid: ReID) -> List[Track]:
    """Read the video once, crop golden detections, and compute embeddings.

    Returns a list of `Track` aligned to `gold_tracks` with embeddings filled.
    """
    # Build frame -> items mapping
    frames_to_items: Dict[int, List[Tuple[int, int, List[int]]]] = {}
    for ti, t in enumerate(gold_tracks):
        for di, det in enumerate(t.detections):
            frames_to_items.setdefault(int(det.frame_idx), []).append((ti, di, det.bbox))

    features_by_track: List[List[np.ndarray | None]] = [[None for _ in t.detections] for t in gold_tracks]

    with VideoReader(video_path) as reader:
        props = reader.get_properties()
        width, height = int(props.width), int(props.height)
        for frame_idx, frame in reader.read_frames():
            items = frames_to_items.get(int(frame_idx))
            if not items:
                continue
            crops: List[np.ndarray] = []
            meta_indices: List[Tuple[int, int]] = []
            for ti, di, bbox in items:
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(int(x1), width - 1))
                y1 = max(0, min(int(y1), height - 1))
                x2 = max(0, min(int(x2), width))
                y2 = max(0, min(int(y2), height))
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0 or (y2 - y1) <= 0 or (x2 - x1) <= 0:
                    crops.append(np.zeros((1, 1, 3), dtype=np.uint8))
                else:
                    crops.append(crop)
                meta_indices.append((ti, di))

            feats = reid.get_features_for_crops(crops)
            assert feats.shape[0] == len(meta_indices)
            for (ti, di), feat in zip(meta_indices, feats):
                features_by_track[ti][di] = l2_normalize(feat)

    # Fill missing with zeros if any and convert to Track objects
    feature_dim: int = 128
    for per_track in features_by_track:
        for e in per_track:
            if e is not None:
                feature_dim = int(e.shape[0])
                break
        else:
            continue
        break

    out_tracks: List[Track] = []
    for tl, per_track_feats in zip(gold_tracks, features_by_track):
        dets: List[Detection] = []
        for det_lite, emb in zip(tl.detections, per_track_feats):
            if emb is None:
                emb = np.zeros((feature_dim,), dtype=np.float32)
            x1, y1, x2, y2 = det_lite.bbox
            dets.append(
                Detection(
                    bbox=BoundingBox(int(x1), int(y1), int(x2), int(y2)),
                    embedding=np.asarray(emb, dtype=np.float32),
                    confidence=float(det_lite.confidence),
                    frame_idx=int(det_lite.frame_idx),
                )
            )
        out_tracks.append(Track(track_id=int(tl.track_id), sorted_detections=dets))
    return out_tracks


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


def _postprocess(tracks: List[Track], props: VideoInfo, transforms: List[Transform]) -> List[Track]:
    stages = [TrackFiltering(), TrackInterpolation(), TrackSmoothing(), TrackRelabeling()]
    out = tracks
    for s in stages:
        out = s.track(out, props, transforms)
    return out


# ─────────────────────────────────── CLI ─────────────────────────────────── #


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare multiple trackers against golden associations (pairwise F1).')
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory with *.golden.tracks.pkl files')
    parser.add_argument('--reid', type=str, default=REID_MODEL_TYPE, choices=['color_hist', 'osnet', 'vit'])
    parser.add_argument(
        '--trackers', type=str, default='botsort,discrete_opt,iter_ilp,oc_sort', help='Comma list of trackers to run'
    )

    args = parser.parse_args()
    trackers = [s.strip() for s in args.trackers.split(',') if s.strip()]

    golden_paths = sorted(Path(args.golden_dir).glob('*.golden.tracks.pkl'))
    if not golden_paths:
        print('No golden files found in the specified directory.')
        return

    reid_model = _init_reid_model(args.reid)

    per_tracker_scores: Dict[str, List[Tuple[str, Dict[str, float]]]] = {name: [] for name in trackers}
    per_tracker_times: Dict[str, List[Tuple[str, float]]] = {name: [] for name in trackers}

    for gpath in golden_paths:
        meta = _load_golden(gpath)
        video_path = Path(meta.input_video_path)
        props = get_video_properties(video_path)

        # Compute embeddings for gold detections so all trackers share the same inputs
        gold_tracks_lite: List[TrackLite] = list(meta.tracks)
        input_tracks: List[Track] = _extract_embeddings_for_gold(video_path.as_posix(), gold_tracks_lite, reid_model)
        actual_input_tracks = []
        for t in input_tracks:
            for d in t.sorted_detections:
                actual_input_tracks.append(Track(track_id=len(actual_input_tracks), sorted_detections=[d]))
        input_tracks = actual_input_tracks

        # Camera transforms
        transforms = compute_stabilization_transforms_gmc(video_path.as_posix())

        # Gold assignment map (by raw gold detections)
        gold_assign = _build_assignment_from_metadata(meta)

        # Build per-tracker
        tracker_by_name: Dict[str, Tracker] = {
            'botsort': BotSortTracker(vid_file_path=video_path.as_posix()),
            'discrete_opt': DiscreteOptTracker(),
            'iter_ilp': IterativeILPTracker(),
            'oc_sort': OCSortEmbedTracker(),
        }

        for name in trackers:
            assert name in tracker_by_name, f'Unknown tracker: {name}'
            current_input_tracks = list(input_tracks)
            if name != 'botsort' and name != 'oc_sort':
                current_input_tracks = Preprocessor().track(current_input_tracks, props)

            start_time = time.time()
            pred_tracks = tracker_by_name[name].track(current_input_tracks, props, transforms)
            per_tracker_times[name].append((gpath.name, time.time() - start_time))

            # TODO reenable? pred_tracks = _postprocess(pred_tracks, props, transforms)
            pred_assign = _build_assignment_from_tracks(pred_tracks)

            # Ensure key alignment with gold: if keys differ, skip evaluation for this video
            if set(gold_assign.keys()) != set(pred_assign.keys()):
                # Attempt to align by fallback: use gold keys and map missing preds to unique ids
                missing = [k for k in gold_assign.keys() if k not in pred_assign]
                if missing:
                    next_tid = (max(pred_assign.values()) + 1) if pred_assign else 1
                    for k in missing:
                        pred_assign[k] = next_tid
                        next_tid += 1

            if set(gold_assign.keys()) != set(pred_assign.keys()):
                print(
                    f'[warn] Key mismatch for {gpath.name} with tracker {name}; skipping F1 computation for this video.'
                )
                continue

            s = _pairwise_scores(gold_assign, pred_assign)
            per_tracker_scores[name].append((gpath.name, s))
            print(f'{gpath.name} {name} {s}')

    # Aggregate and print summary
    print('\nPairwise F1 by tracker:')
    for name, items in per_tracker_scores.items():
        if not items:
            print(f'  {name}: n/a')
            continue
        avg_f1 = sum(s['pairwise_f1'] for _, s in items) / len(items)
        avg_time = sum(t for _, t in per_tracker_times[name]) / len(per_tracker_times[name])
        print(f'  {name}: {avg_f1:.4f}  (n={len(items)}) {avg_time:.2f}s')


if __name__ == '__main__':
    main()
