from __future__ import annotations

import sys
import pickle
import argparse
from pathlib import Path


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from server.inference.src.util.video_io import get_video_properties
from server.inference.src.tracking.detector import SurferDetector
from server.inference.src.tracking.preprocessing.preprocessor import TrackPreProcessor
from server.inference.src.tracking.greedy_tracker import GreedyTracker
from server.inference.src.tracking.discrete_opt_tracker import DiscreteILPTracker
from server.inference.src.common_types import Detection, Track
from server.inference.src.player.core.player_state import Metadata, TrackLite, DetectionLite
from server.inference.src.settings import YOLO_MODEL_PATH


def _detections_to_initial_tracks(detections: list[Detection]) -> list[Track]:
    return [Track(track_id=i + 1, sorted_detections=[det]) for i, det in enumerate(detections)]


detector = SurferDetector(yolo_model_path=YOLO_MODEL_PATH)


def _run_pipeline(video_path: Path, tracker_name: str) -> list[Track]:
    video_props = get_video_properties(video_path)
    detections = detector.run_object_detection_on_video(video_path.as_posix())

    tracks: list[Track] = _detections_to_initial_tracks(detections)
    tracks = TrackPreProcessor().track(tracks, video_props)

    if tracker_name == 'none':
        return tracks
    if tracker_name == 'greedy':
        return GreedyTracker().track(tracks, video_props)
    if tracker_name == 'ilp':
        return DiscreteILPTracker().track(tracks, video_props)
    raise ValueError(f'Unknown tracker: {tracker_name}')


def _load_golden(path: Path) -> Metadata:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, Metadata):
        raise TypeError('Golden file does not contain Metadata')
    return data


def _build_assignment_from_tracks(tracks: list[Track]) -> dict[tuple[int, int, int, int, int], int]:
    """Map detection identity -> cluster id.

    Identity is (frame_idx, x1, y1, x2, y2) using integer coordinates.
    """
    assignment: dict[tuple[int, int, int, int, int], int] = {}
    for t in tracks:
        for det in t.sorted_detections:
            k = (int(det.frame_idx), int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2))
            assignment[k] = int(t.track_id)
    return assignment


def _build_assignment_from_metadata(meta: Metadata) -> dict[tuple[int, int, int, int, int], int]:
    assignment: dict[tuple[int, int, int, int, int], int] = {}
    for t in meta.tracks:
        assert isinstance(t, TrackLite)
        for det in t.detections:
            assert isinstance(det, DetectionLite)
            x1, y1, x2, y2 = det.bbox
            k = (int(det.frame_idx), int(x1), int(y1), int(x2), int(y2))
            assignment[k] = int(t.track_id)
    return assignment


def _pairwise_scores(gold: dict, pred: dict) -> dict[str, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a tracker against golden tracklet associations.')
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory containing golden .golden.tracks.pkl')
    parser.add_argument('--tracker', type=str, default='greedy', choices=['none', 'greedy', 'ilp'])

    args = parser.parse_args()

    # Aggregate metrics
    metrics_list = []
    for golden_path in Path(args.golden_dir).glob('*.golden.tracks.pkl'):
        gold_meta = _load_golden(golden_path)
        pred_tracks = _run_pipeline(Path(gold_meta.input_video_path), args.tracker)
        gold_assign = _build_assignment_from_metadata(gold_meta)
        pred_assign = _build_assignment_from_tracks(pred_tracks)

        # Require same keys; if not, skip and warn
        if set(gold_assign.keys()) != set(pred_assign.keys()):
            print(f'SKIP: key mismatch for {golden_path.name} (detections differ).')
            continue

        scores = _pairwise_scores(gold_assign, pred_assign)
        metrics_list.append((golden_path.name, scores))

    if not metrics_list:
        print('No comparable videos evaluated (missing goldens or key mismatches).')
        return

    # Print per-video and averages
    print('Per-video evaluation (pairwise over detections):')
    for name, s in metrics_list:
        print(f'- {name}')
        print(f'  detections: {int(s["num_detections"])}, pairs: {int(s["pairs"])}')
        print(
            f'  precision : {s["pairwise_precision"]:.4f}, recall: {s["pairwise_recall"]:.4f}, f1: {s["pairwise_f1"]:.4f}'
        )
        print(f'  rand_idx  : {s["rand_index"]:.4f}, jaccard: {s["jaccard_same"]:.4f}')

    # Averages (simple mean over videos)
    def _avg(key: str) -> float:
        vals = [s[key] for _, s in metrics_list]
        return sum(vals) / max(1, len(vals))

    print('Averages:')
    print(f'  videos    : {len(metrics_list)}')
    print(f'  precision : {_avg("pairwise_precision"):.4f}')
    print(f'  recall    : {_avg("pairwise_recall"):.4f}')
    print(f'  f1        : {_avg("pairwise_f1"):.4f}')
    print(f'  rand_idx  : {_avg("rand_index"):.4f}')
    print(f'  jaccard   : {_avg("jaccard_same"):.4f}')


if __name__ == '__main__':
    main()
