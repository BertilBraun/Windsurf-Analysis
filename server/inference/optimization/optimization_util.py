from __future__ import annotations

from dataclasses import dataclass
import sys
import optuna
import argparse
import numpy as np

from pathlib import Path
from functools import cache
from typing import Callable, Dict, Generator, List, Literal, Tuple, Optional


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from server.inference.src.util.similarity_helpers import Embedding
from server.inference.src.player.core.player_state import Metadata, TrackLite
from server.inference.src.common_types import BoundingBox, Detection, Track
from server.inference.src.util.video_io import VideoReader
from server.inference.src.tracking.reid import ReID, ReIDColorHistogram, ReIDOSNet, ReIDViT
from server.inference.src.settings import REID_MODEL_TYPE, OSNET_REID_MODEL_PATH


# ----------------------------- Built-in configuration ----------------------------- #

# Global constants for optimization and sampling. Adjust here as needed.
TRIALS: int = 200
RANDOM_SEED: int = 42


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


def _extract_embeddings_for_tracklets(video_path: str, tracklets: List[TrackLite]) -> List[List[Embedding]]:
    """Stream the video once and compute embeddings for each detection in the given tracklets.

    Returns a list parallel to `tracklets`, where each item is a list of embeddings aligned 1:1 with detections.
    """
    # Build frame -> items mapping
    frames_to_items: dict[int, List[Tuple[int, int, BoundingBox]]] = {}
    for track_id, t in enumerate(tracklets):
        for detection_index, detection in enumerate(t.detections):
            frames_to_items.setdefault(int(detection.frame_idx), []).append(
                (
                    track_id,
                    detection_index,
                    BoundingBox(detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3]),
                )
            )

    embeddings_by_tracklet: List[List[Optional[Embedding]]] = [[None for _ in t.detections] for t in tracklets]

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
            for track_id, detection_index, bbox in pending:
                x1, y1, x2, y2 = bbox.clamp(0, 0, width, height)
                crop = frame[y1:y2, x1:x2]
                if crop is None or crop.size == 0 or (y2 - y1) <= 0 or (x2 - x1) <= 0:
                    # Skip invalid crops; will fill zero vector later
                    meta_indices.append((track_id, detection_index))
                    crops.append(np.zeros((1, 1, 3), dtype=np.uint8))
                else:
                    meta_indices.append((track_id, detection_index))
                    crops.append(crop)

            feats = reid.get_features_for_crops(crops)
            assert len(feats) == len(meta_indices)
            for (track_id, detection_index), feat in zip(meta_indices, feats):
                embeddings_by_tracklet[track_id][detection_index] = feat

    finalized: List[List[Embedding]] = []
    for per_track in embeddings_by_tracklet:
        out_track: List[Embedding] = []
        for e in per_track:
            assert e is not None
            out_track.append(e)
        finalized.append(out_track)
    return finalized


def _to_track_with_embeddings(tl: TrackLite, embeddings: List[Embedding]) -> Track:
    dets: List[Detection] = []
    for det_lite, emb in zip(tl.detections, embeddings):
        x1, y1, x2, y2 = det_lite.bbox
        dets.append(
            Detection(
                bbox=BoundingBox(int(x1), int(y1), int(x2), int(y2)),
                embedding=emb,
                confidence=float(det_lite.confidence),
                frame_idx=int(det_lite.frame_idx),
            )
        )
    return Track(track_id=int(tl.track_id), sorted_detections=dets)


@cache
def load_full_tracks(golden_path: Path) -> Tuple[List[Track], Metadata]:
    meta = _load_golden(golden_path)
    video_path = meta.input_video_path
    embeddings = _extract_embeddings_for_tracklets(video_path, list(meta.tracks))
    full_tracks = [_to_track_with_embeddings(tl, emb) for tl, emb in zip(meta.tracks, embeddings)]
    return full_tracks, meta


def each_golden(golden_dir: Path | str) -> Generator[Tuple[List[Track], Metadata], None, None]:
    golden_paths = sorted(Path(golden_dir).glob('*.golden.tracks.pkl'))
    if not golden_paths:
        print('No golden files found in the specified directory.')
        return

    for golden_path in golden_paths:
        yield load_full_tracks(golden_path)


@dataclass(frozen=True)
class AssignmentKey:
    frame_idx: int
    x1: int
    y1: int
    x2: int
    y2: int

    def __lt__(self, other: AssignmentKey) -> bool:
        return (self.frame_idx, self.x1, self.y1, self.x2, self.y2) < (
            other.frame_idx,
            other.x1,
            other.y1,
            other.x2,
            other.y2,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AssignmentKey):
            return False
        return (self.frame_idx, self.x1, self.y1, self.x2, self.y2) == (
            other.frame_idx,
            other.x1,
            other.y1,
            other.x2,
            other.y2,
        )

    def __hash__(self) -> int:
        return hash((self.frame_idx, self.x1, self.y1, self.x2, self.y2))


def build_assignment_from_tracks(tracks: List[Track]) -> Dict[AssignmentKey, int]:
    assignment: Dict[AssignmentKey, int] = {}
    for t in tracks:
        for det in t.sorted_detections:
            k = AssignmentKey(
                frame_idx=int(det.frame_idx),
                x1=int(det.bbox.x1),
                y1=int(det.bbox.y1),
                x2=int(det.bbox.x2),
                y2=int(det.bbox.y2),
            )
            assignment[k] = int(t.track_id)
    return assignment


def build_assignment_from_metadata(meta: Metadata) -> Dict[AssignmentKey, int]:
    assignment: Dict[AssignmentKey, int] = {}
    for t in meta.tracks:
        for det in t.detections:
            x1, y1, x2, y2 = det.bbox
            k = AssignmentKey(frame_idx=int(det.frame_idx), x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
            assignment[k] = int(t.track_id)
    return assignment


@dataclass(frozen=True)
class PairwiseScores:
    num_detections: float
    pairs: float
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    rand_index: float
    jaccard_same: float


def pairwise_scores(gold: Dict[AssignmentKey, int], pred: Dict[AssignmentKey, int]) -> PairwiseScores:
    assert set(gold.keys()) == set(pred.keys()), 'Gold and pred keys must be the same'
    keys = sorted(gold.keys())
    n = len(keys)
    if n < 2:
        return PairwiseScores(
            num_detections=float(n),
            pairs=0.0,
            pairwise_precision=1.0,
            pairwise_recall=1.0,
            pairwise_f1=1.0,
            rand_index=1.0,
            jaccard_same=1.0,
        )

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
    return PairwiseScores(
        num_detections=float(n),
        pairs=float(pairs),
        pairwise_precision=float(prec),
        pairwise_recall=float(rec),
        pairwise_f1=float(f1),
        rand_index=float(rand),
        jaccard_same=float(jaccard_same),
    )


def optimize(
    objective: Callable[[optuna.trial.Trial], float],
    direction: Literal['minimize', 'maximize'] = 'minimize',
    trials: int = TRIALS,
    seed: int = RANDOM_SEED,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=max(1, trials), show_progress_bar=True)

    print('\nBest params (objective = %.6f):' % float(study.best_value))
    best = dict(study.best_trial.params)
    for k in sorted(best.keys()):
        print(f'  {k}: {best[k]}')

    return study


def main() -> None:
    parser = argparse.ArgumentParser(description='Optimize built-in tracklet matching scorer on golden tracklets.')
    parser.add_argument('--golden-dir', type=str, required=True, help='Directory with *.golden.tracks.pkl files')

    args = parser.parse_args()

    def objective(trial: optuna.trial.Trial) -> float:
        ema = trial.suggest_float('ema', 0.0, 1.0)

        scores = []
        for tracks, meta in each_golden(args.golden_dir):
            score = sum(a.mean_embedding(ema=ema).distance(b.mean_embedding(ema=ema)) for a, b in zip(tracks, tracks))
            scores.append(score)

        return sum(scores) / len(scores)

    optimize(objective, direction='maximize', trials=100, seed=42)


if __name__ == '__main__':
    main()
