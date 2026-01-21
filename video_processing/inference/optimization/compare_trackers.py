from __future__ import annotations

import sys
import argparse
from pathlib import Path
import time
from typing import Dict, List, Tuple


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from video_processing.inference.src.tracking.tracking import Tracker
from video_processing.inference.src.common_types import Track
from video_processing.inference.src.util.video_io import get_video_properties
from video_processing.inference.src.settings import REID_MODEL_TYPE
from video_processing.inference.src.tracking.discrete_opt_tracker import DiscreteOptTracker
from video_processing.inference.src.tracking.iterative_ilp_tracker import IterativeILPTracker
from video_processing.inference.src.tracking.oc_sort import OCSortEmbedTracker
from video_processing.inference.src.tracking.preprocessing.preprocessor import TrackPreProcessor
from video_processing.inference.src.visualization.stabilize import compute_stabilization_transforms_gmc

from video_processing.inference.optimization.optimization_util import (
    PairwiseScores,
    each_golden,
    build_assignment_from_metadata,
    build_assignment_from_tracks,
    pairwise_scores,
)

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

    per_tracker_scores: Dict[str, List[Tuple[str, PairwiseScores]]] = {name: [] for name in trackers}
    per_tracker_times: Dict[str, List[Tuple[str, float]]] = {name: [] for name in trackers}

    for tracks, meta in each_golden(args.golden_dir):
        video_path = Path(meta.input_video_path)
        props = get_video_properties(video_path)

        # Compute embeddings for gold detections so all trackers share the same inputs
        input_tracks: List[Track] = tracks
        actual_input_tracks = []
        for t in input_tracks:
            for d in t.sorted_detections:
                actual_input_tracks.append(Track(track_id=len(actual_input_tracks), sorted_detections=[d]))
        input_tracks = actual_input_tracks

        # Camera transforms
        transforms = compute_stabilization_transforms_gmc(video_path.as_posix())

        # Gold assignment map (by raw gold detections)
        gold_assign = build_assignment_from_metadata(meta)

        # Build per-tracker
        tracker_by_name: Dict[str, Tracker] = {
            'discrete_opt': DiscreteOptTracker(),
            'iter_ilp': IterativeILPTracker(video_path.as_posix()),
            'oc_sort': OCSortEmbedTracker(),
            'preprocessor': TrackPreProcessor(),
        }
        requires_preprocessor = ['discrete_opt', 'iter_ilp']

        for name in trackers:
            assert name in tracker_by_name, f'Unknown tracker: {name}'
            current_input_tracks = list(input_tracks)
            if name in requires_preprocessor:
                current_input_tracks = TrackPreProcessor().track(current_input_tracks, props, transforms)

            start_time = time.time()
            pred_tracks = tracker_by_name[name].track(current_input_tracks, props, transforms)
            per_tracker_times[name].append((video_path.name, time.time() - start_time))

            # TODO reenable? pred_tracks = _postprocess(pred_tracks, props, transforms)
            pred_assign = build_assignment_from_tracks(pred_tracks)

            # Ensure key alignment with gold: if keys differ, skip evaluation for this video
            if set(gold_assign.keys()) != set(pred_assign.keys()):
                # Attempt to align by fallback: use gold keys and map missing preds to unique ids
                missing = [k for k in gold_assign.keys() if k not in pred_assign]
                if missing:
                    next_tid = (max(pred_assign.values()) + 1) if pred_assign else 1
                    for k in missing:
                        pred_assign[k] = next_tid
                        next_tid += 1

            assert set(gold_assign.keys()) == set(pred_assign.keys()), (
                f'Key mismatch for {video_path.name} with tracker {name}'
            )

            s = pairwise_scores(gold_assign, pred_assign)
            per_tracker_scores[name].append((video_path.name, s))
            print(f'{video_path.name} {name} {s}')

    # Aggregate and print summary
    print('\nPairwise F1 by tracker:')
    for name, items in per_tracker_scores.items():
        if not items:
            print(f'  {name}: n/a')
            continue
        avg_f1 = sum(s.f1 for _, s in items) / len(items)
        avg_time = sum(t for _, t in per_tracker_times[name]) / len(per_tracker_times[name])
        print(f'  {name}: {avg_f1:.4f}  (n={len(items)}) {avg_time:.2f}s')


if __name__ == '__main__':
    main()
