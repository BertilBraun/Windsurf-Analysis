from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sys
import pickle
import optuna
import argparse
import numpy as np

from pathlib import Path
from functools import cache
from typing import Any, Callable, Dict, Generator, Generic, List, Literal, Tuple, Optional, TypeVar
import multiprocessing as mp
import traceback


# Ensure project imports work when executed as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from video_processing.inference.src.util.similarity_helpers import Embedding
from video_processing.inference.src.player.core.player_state import Metadata, TrackLite
from video_processing.inference.src.common_types import BoundingBox, Detection, Track, Keypoint, Point
from video_processing.inference.src.util.video_io import VideoReader
from video_processing.inference.src.tracking.detector import init_reid_model
from video_processing.inference.src.settings import REID_MODEL_TYPE


# ----------------------------- Built-in configuration ----------------------------- #

# Global constants for optimization and sampling. Adjust here as needed.
TRIALS: int = 200
RANDOM_SEED: int = 42


class _CompatUnpickler(pickle.Unpickler):
    """
    Allows loading old golden pickle files after project/module refactors.

    Common historical module prefixes:
      - server.inference.*  (backend layout)
    """

    def find_class(self, module: str, name: str):  # type: ignore[override]
        # Handle historical backend prefix (e.g. server.inference.*).
        if module.startswith('server.inference.'):
            suffix = module[len('server.inference.') :]
            return super().find_class(f'video_processing.inference.{suffix}', name)

        return super().find_class(module, name)


def load_pickle_compat(path: Path) -> object:
    with open(path, 'rb') as f:
        return _CompatUnpickler(f).load()


def _load_golden(path: Path) -> Metadata:
    data = load_pickle_compat(path)
    if not isinstance(data, Metadata):
        raise TypeError('Golden file does not contain Metadata')
    return data


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

    reid = init_reid_model(REID_MODEL_TYPE)

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
                boom=Keypoint(point=Point(0, 0), conf=0.0),
                mast_tip=Keypoint(point=Point(0, 0), conf=0.0),
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


def list_golden_paths(golden_dir: Path | str) -> List[Path]:
    """Sorted list of `*.golden.tracks.pkl` paths."""
    return sorted(Path(golden_dir).glob('*.golden.tracks.pkl'))


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
    precision: float
    recall: float
    f1: float
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
            precision=1.0,
            recall=1.0,
            f1=1.0,
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
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        rand_index=float(rand),
        jaccard_same=float(jaccard_same),
    )


@dataclass(frozen=True)
class _WorkItem:
    task_id: int
    path: str
    params: Dict[str, object]


@dataclass(frozen=True)
class _WorkResult:
    task_id: int
    ok: bool
    value: object | None
    error: str | None


def _queue_worker_loop(
    in_q: Any,
    out_q: Any,
    worker_fn: Callable[[str, Dict[str, object]], object],
) -> None:
    while True:
        item = in_q.get()
        if item is None:
            return
        if not isinstance(item, _WorkItem):
            out_q.put(_WorkResult(task_id=-1, ok=False, value=None, error=f'Invalid work item: {type(item)}'))
            continue
        try:
            value = worker_fn(str(item.path), dict(item.params))
            out_q.put(_WorkResult(task_id=int(item.task_id), ok=True, value=value, error=None))
        except BaseException:
            out_q.put(
                _WorkResult(
                    task_id=int(item.task_id),
                    ok=False,
                    value=None,
                    error=traceback.format_exc(),
                )
            )


T = TypeVar('T')


class SharedQueueWorkerPool(Generic[T]):
    """Persistent multiprocessing pool using per-worker pinned queues.

    Designed for Optuna-style loops where the same (path, params) worker is called repeatedly.
    """

    def __init__(self, *, worker_fn: Callable[[str, Dict[str, object]], T], workers: int) -> None:
        self._ctx = mp.get_context('spawn')
        self._out_q = self._ctx.Queue()
        self._processes: list[mp.Process] = []
        self._in_qs: list[Any] = []
        self._closed = False

        self._workers = max(1, int(workers))
        for _ in range(self._workers):
            in_q = self._ctx.Queue()
            self._in_qs.append(in_q)
            p = self._ctx.Process(target=_queue_worker_loop, args=(in_q, self._out_q, worker_fn), daemon=True)
            p.start()
            self._processes.append(p)  # type: ignore

    def _worker_index_for_path(self, path: str) -> int:
        # Deterministic routing to preserve per-process caches (independent of Python's hash randomization).
        digest = hashlib.md5(path.encode('utf-8')).digest()
        return int.from_bytes(digest[:4], byteorder='little', signed=False) % self._workers

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for q in self._in_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self._processes:
            try:
                p.join(timeout=5)
            except Exception:
                pass
        for p in self._processes:
            if p.is_alive():
                try:
                    p.kill()
                except Exception:
                    pass

    def __enter__(self) -> 'SharedQueueWorkerPool':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def map_paths(self, paths: List[str], params: Dict[str, object]) -> List[T]:
        if not paths:
            return []
        # Enqueue all work for this batch.
        for task_id, p in enumerate(paths):
            idx = self._worker_index_for_path(str(p))
            self._in_qs[idx].put(_WorkItem(task_id=int(task_id), path=str(p), params=dict(params)))

        # Collect exactly one result per task_id.
        results: list[T | None] = [None for _ in range(len(paths))]
        first_error: tuple[int, str] | None = None
        for _ in range(len(paths)):
            res = self._out_q.get()
            if not isinstance(res, _WorkResult):
                first_error = first_error or (-1, f'Invalid work result: {type(res)}')
                continue
            if not res.ok:
                first_error = first_error or (int(res.task_id), str(res.error or 'unknown error'))
                continue
            if 0 <= int(res.task_id) < len(results):
                results[int(res.task_id)] = res.value  # type: ignore

        if first_error is not None:
            task_id, err = first_error
            path_hint = paths[task_id] if 0 <= task_id < len(paths) else '<unknown>'
            raise RuntimeError(f'Worker failed for task_id={task_id} path={path_hint}:\n{err}')

        # At this point all slots should be filled.
        assert all(r is not None for r in results)
        return [r for r in results if r is not None]


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
