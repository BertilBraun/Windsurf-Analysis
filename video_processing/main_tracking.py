from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import modal

from inference.src.common_types import BoundingBox, Detection, FrameIndex, Track
from inference.src.settings import REID_MODEL_TYPE
from inference.src.tracking.detector import EmbeddingExtractor, RawDetection
from inference.src.tracking.iterative_ilp_tracker import IterativeILPTracker
from inference.src.tracking.preprocessing.preprocessor import TrackPreProcessor
from inference.src.tracking.track_processing import TrackPostProcessing
from inference.src.tracking.tracking import Tracker
from inference.src.util.video_io import VideoReader, get_video_properties
from inference.src.util.timing import timeit
from inference.src.visualization.stabilize import Transform, vidstab_like_transforms

from main_inference import (
    report_job_failure_on_exception,
    image as inference_image,
    send_complete,
    send_progress,
    volume as shared_volume,
    wait_for_volume_reload,
)


app = modal.App('windsurf-analysis-tracking', image=inference_image)


def clamp_percentage(p: float) -> float:
    if p < -1 or p > 2:
        print(f'WARNING: Clamping percentage {p} to 0-1')
    return max(0, min(p, 1))


@app.function(
    secrets=[modal.Secret.from_name('backend-secret')],
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
        input_video_path = f'/data/{job_id}_upright.mp4'
        wait_for_volume_reload(input_video_path)

        send_progress(job_id, 'appearance')

        parsed_transforms = [Transform(**transform) for transform in transforms]
        with timeit(f'{job_id}: Extracting detections'):
            detections = _extract_detections(raw_detections, input_video_path)

        props = get_video_properties(input_video_path)

        send_progress(job_id, 'tracking')

        with timeit(f'{job_id}: Trackers'):
            trackers: Sequence[Tracker] = [
                TrackPreProcessor(),
                IterativeILPTracker(),
                TrackPostProcessing(),
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
                'start_time_seconds': t.start_frame / props.fps,
                'duration_seconds': t.duration_frames / props.fps,
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

        with timeit(f'{job_id}: Stabilization Optimization'):
            # Compute the transforms which the frontend should use as per frame warps for stabilization
            # stabilized_transforms = optimize_trajectory_world(transforms, properties.width, properties.height)
            smoothing_window = min(20, props.total_frames - 1)
            stabilized_transforms = vidstab_like_transforms(parsed_transforms, smoothing_window)

        # Convert transforms payload into result with time_percent
        stabilization_transforms = [
            {
                'time_percent': clamp_percentage(
                    i / (len(stabilized_transforms) + 1)
                ),  # NOTE +1 because we have one transform less than the number of frames
                'dx': t.dx,
                'dy': t.dy,
                'da': t.da,
            }
            for i, t in enumerate(stabilized_transforms)
        ]

        results = {
            'tracks': tracks,
            'dominant_orientation': dominant_orientation,
            'stabilization_transforms': stabilization_transforms,
        }

        send_complete(job_id, 'succeeded', results)


def _extract_detections(raw_detections: list[dict], input_video_path: str) -> list[Detection]:
    # Extract detections from raw detections
    detections_by_frame: dict[FrameIndex, list[tuple[BoundingBox, float]]] = defaultdict(list)
    for detection in raw_detections:
        detections_by_frame[detection['frame_idx']].append(
            (
                BoundingBox(
                    x1=int(detection['bbox'][0]),
                    y1=int(detection['bbox'][1]),
                    x2=int(detection['bbox'][2]),
                    y2=int(detection['bbox'][3]),
                ),
                float(detection['confidence']),
            )
        )

    raw_detections_with_crops: list[RawDetection] = []
    with VideoReader(input_video_path) as reader:
        for frame_idx, frame in reader.read_frames():
            for bbox, confidence in detections_by_frame[frame_idx]:
                raw_detections_with_crops.append(
                    RawDetection(
                        bbox=bbox,
                        confidence=confidence,
                        frame_idx=frame_idx,
                        crop=frame[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2],
                    )
                )

    return EmbeddingExtractor(REID_MODEL_TYPE).run_embedding_pass(raw_detections_with_crops)
