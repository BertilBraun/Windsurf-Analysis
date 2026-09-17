from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from boxmot import BotSort, OcSort

from video_processing.inference.optimization.association_benchmark_data import (
    GoldenSequence,
    materialize_original_detections,
    observation_key,
    singleton_tracks,
)
from video_processing.inference.optimization.optimization_util import AssignmentKey
from video_processing.inference.src.common_types import Track
from video_processing.inference.src.player.core.player_state import DetectionLite
from video_processing.inference.src.tracking.ilp_tracker import ILPTracker
from video_processing.inference.src.tracking.preprocessing.preprocessor import TrackPreProcessor
from video_processing.inference.src.util.video_io import VideoReader, get_video_properties
from video_processing.inference.src.visualization.stabilize import compute_stabilization_transforms_masked_vidstab


class BenchmarkTracker(str, Enum):
    OC_SORT = 'boxmot_oc_sort'
    BOT_SORT = 'boxmot_bot_sort_gmc_no_reid'
    PRODUCTION = 'production_preprocessor_ilp'


@dataclass(frozen=True)
class TrackerPrediction:
    tracker: BenchmarkTracker
    assignment: dict[AssignmentKey, int]
    runtime_seconds: float


def _build_unique_assignment(tracks: list[Track]) -> dict[AssignmentKey, int]:
    assignment: dict[AssignmentKey, int] = {}
    for track in tracks:
        for detection in track.sorted_detections:
            key = observation_key(detection)
            if key in assignment:
                raise ValueError(f'Observation {key} is owned by more than one predicted track.')
            assignment[key] = int(track.track_id)
    return assignment


def _detections_by_frame(sequence: GoldenSequence) -> dict[int, list[DetectionLite]]:
    by_frame: dict[int, list[DetectionLite]] = defaultdict(list)
    for track in sequence.original_tracks:
        for detection in track.detections:
            by_frame[int(detection.frame_idx)].append(detection)
    for detections in by_frame.values():
        detections.sort(key=observation_key)
    return by_frame


def _boxmot_input(detections: list[DetectionLite]) -> np.ndarray:
    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(
        [[*detection.bbox, detection.confidence, 0.0] for detection in detections],
        dtype=np.float32,
    )


def _record_boxmot_outputs(
    outputs: np.ndarray,
    frame_detections: list[DetectionLite],
    assignment: dict[AssignmentKey, int],
) -> None:
    if outputs.size == 0:
        return
    for output in outputs:
        detection_index = int(round(float(output[-1])))
        if detection_index < 0 or detection_index >= len(frame_detections):
            raise ValueError(f'BoxMOT returned invalid detection index {detection_index}.')
        key = observation_key(frame_detections[detection_index])
        if key in assignment:
            raise ValueError(f'BoxMOT returned observation {key} more than once.')
        assignment[key] = int(round(float(output[4])))


def run_oc_sort(sequence: GoldenSequence) -> TrackerPrediction:
    tracker = OcSort(
        per_class=False,
        min_conf=0.1,
        det_thresh=0.2,
        max_age=30,
        min_hits=3,
        asso_threshold=0.3,
        delta_t=3,
        asso_func='iou',
        inertia=0.2,
        use_byte=False,
        Q_xy_scaling=0.01,
        Q_s_scaling=0.0001,
    )
    detections_by_frame = _detections_by_frame(sequence)
    properties = sequence.metadata.video_properties
    blank_frame = np.zeros((properties.height, properties.width, 3), dtype=np.uint8)
    assignment: dict[AssignmentKey, int] = {}
    started_at = perf_counter()
    for frame_index in range(properties.total_frames):
        frame_detections = detections_by_frame.get(frame_index, [])
        outputs = tracker.update(_boxmot_input(frame_detections), blank_frame)
        _record_boxmot_outputs(outputs, frame_detections, assignment)
    return TrackerPrediction(BenchmarkTracker.OC_SORT, assignment, perf_counter() - started_at)


def run_bot_sort(sequence: GoldenSequence) -> TrackerPrediction:
    tracker = BotSort(
        reid_weights=Path('unused.pt'),
        device=torch.device('cpu'),
        half=False,
        per_class=False,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        cmc_method='ecc',
        frame_rate=int(round(sequence.metadata.video_properties.fps)),
        fuse_first_associate=False,
        with_reid=False,
    )
    detections_by_frame = _detections_by_frame(sequence)
    assignment: dict[AssignmentKey, int] = {}
    started_at = perf_counter()
    with VideoReader(sequence.video_path) as reader:
        for frame_index, frame in reader.read_frames():
            frame_detections = detections_by_frame.get(frame_index, [])
            outputs = tracker.update(_boxmot_input(frame_detections), frame)
            _record_boxmot_outputs(outputs, frame_detections, assignment)
    return TrackerPrediction(BenchmarkTracker.BOT_SORT, assignment, perf_counter() - started_at)


def run_production(sequence: GoldenSequence) -> TrackerPrediction:
    detections = materialize_original_detections(sequence)
    bboxes_by_frame: dict[int, list[list[int]]] = defaultdict(list)
    for detection in detections:
        bboxes_by_frame[int(detection.frame_idx)].append(
            [int(detection.bbox.x1), int(detection.bbox.y1), int(detection.bbox.x2), int(detection.bbox.y2)]
        )

    started_at = perf_counter()
    transforms = compute_stabilization_transforms_masked_vidstab(
        sequence.video_path,
        bboxes_by_frame=bboxes_by_frame,
        mask_margin_px=20,
    )
    video_properties = get_video_properties(sequence.video_path)
    fragments = TrackPreProcessor().track(singleton_tracks(detections), video_properties, transforms)
    tracks = ILPTracker(str(sequence.video_path)).track(fragments, video_properties, transforms)
    assignment = _build_unique_assignment(tracks)
    extra_keys = set(assignment) - set(sequence.gold_assignment)
    if extra_keys:
        raise ValueError(f'Production tracker changed {len(extra_keys)} observation keys for {sequence.name}.')
    return TrackerPrediction(BenchmarkTracker.PRODUCTION, assignment, perf_counter() - started_at)


def run_tracker(tracker: BenchmarkTracker, sequence: GoldenSequence) -> TrackerPrediction:
    match tracker:
        case BenchmarkTracker.OC_SORT:
            return run_oc_sort(sequence)
        case BenchmarkTracker.BOT_SORT:
            return run_bot_sort(sequence)
        case BenchmarkTracker.PRODUCTION:
            return run_production(sequence)
