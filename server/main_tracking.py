from __future__ import annotations

import os
from typing import Sequence

import modal

from server.inference.src.common_types import BoundingBox, Detection, FrameIndex, Track
from server.inference.src.settings import REID_MODEL_TYPE
from server.inference.src.tracking.detector import EmbeddingExtractor, RawDetection
from server.inference.src.tracking.iterative_ilp_tracker import IterativeILPTracker
from server.inference.src.tracking.preprocessing.preprocessor import Preprocessor
from server.inference.src.tracking.track_processing import TrackRelabeling
from server.inference.src.tracking.tracking import Tracker
from server.inference.src.util.video_io import VideoReader, get_video_properties

from .inference.src.util.timing import timeit
from .inference.src.visualization.stabilize import Transform
from .main_inference import (
    report_job_failure_on_exception,
    image as inference_image,
    send_complete,
    send_progress,
    volume as shared_volume,
)


app = modal.App('windsurf-analysis-tracking', image=inference_image)


def clamp_percentage(p: float) -> float:
    return max(0, min(p, 1))


@app.function(
    volumes={'/data': shared_volume},
    scaledown_window=10,
    cpu=2.0,
)
@modal.concurrent(max_inputs=16, target_inputs=12)
def embedding_extraction_and_tracking(
    job_id: str,
    dominant_orientation: int,
    transforms: list[dict],
    raw_detections: list[dict],
):
    with report_job_failure_on_exception(job_id):
        shared_volume.reload()

        input_video_path = f'/data/{job_id}.mp4'
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f'Input video not found: {input_video_path}')

        send_progress(job_id, 'appearance')

        parsed_transforms = [Transform(**transform) for transform in transforms]
        with timeit(f'{job_id}: Extracting detections'):
            detections = _extract_detections(raw_detections, input_video_path)

        props = get_video_properties(input_video_path)

        send_progress(job_id, 'tracking')

        with timeit(f'{job_id}: Trackers'):
            trackers: Sequence[Tracker] = [
                Preprocessor(),
                IterativeILPTracker(),
                # TrackInterpolation(), # Not needed - done in frontend
                # TrackSmoothing(), # Not needed - done in frontend
                TrackRelabeling(),
            ]
            processed_tracks = [
                Track(track_id=i, sorted_detections=[detection]) for i, detection in enumerate(detections)
            ]
            for tracker in trackers:
                processed_tracks = tracker.track(processed_tracks, props, parsed_transforms)

        print(f'{job_id}: Found {len(processed_tracks)} tracks')

        tracks = [
            {
                'track_id': t.track_id,
                'start_percent': clamp_percentage(t.start_frame / props.total_frames),
                'end_percent': clamp_percentage(t.end_frame / props.total_frames),
                'start_time_seconds': clamp_percentage(t.start_frame / props.fps),
                'duration_seconds': clamp_percentage(t.duration_frames / props.fps),
                'detections': [
                    {
                        'time_percent': clamp_percentage(d.frame_idx / props.total_frames),
                        'bbox': [
                            clamp_percentage(d.bbox.x1 / props.width),
                            clamp_percentage(d.bbox.y1 / props.height),
                            clamp_percentage(d.bbox.x2 / props.width),
                            clamp_percentage(d.bbox.y2 / props.height),
                        ],
                        'confidence': clamp_percentage(d.confidence),
                    }
                    for d in t.sorted_detections
                ],
            }
            for t in processed_tracks
        ]

        # Convert transforms payload into result with time_percent
        stabilization_transforms = [
            {
                'time_percent': clamp_percentage(i / len(parsed_transforms)),
                'dx': t.dx,
                'dy': t.dy,
                'da': t.da,
            }
            for i, t in enumerate(parsed_transforms)
        ]

        results = {
            'tracks': tracks,
            'dominant_orientation': dominant_orientation,
            'stabilization_transforms': stabilization_transforms,
        }

        send_complete(job_id, 'succeeded', results)


def _extract_detections(raw_detections: list[dict], input_video_path: str) -> list[Detection]:
    # Extract detections from raw detections
    detections_by_frame: dict[FrameIndex, list[tuple[BoundingBox, float]]] = {}
    for detection in raw_detections:
        detections_by_frame[detection['frame_idx']].append((BoundingBox(**detection['bbox']), detection['confidence']))

    raw_detections_with_crops: list[RawDetection] = []
    with VideoReader(input_video_path) as reader:
        for frame_idx, frame in reader.read_frames():
            for bbox, confidence in detections_by_frame.get(frame_idx, []):
                raw_detections_with_crops.append(
                    RawDetection(
                        bbox=bbox,
                        confidence=confidence,
                        frame_idx=frame_idx,
                        crop=frame[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2],
                    )
                )

    return EmbeddingExtractor(REID_MODEL_TYPE).run_embedding_pass(raw_detections_with_crops)
