from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import modal

from inference.src.common_types import BoundingBox, Track
from inference.src.settings import REID_MODEL_TYPE
from inference.src.tracking.detector import EmbeddingExtractor, RawDetection
from inference.src.tracking.iterative_ilp_tracker import IterativeILPTracker
from inference.src.tracking.preprocessing.preprocessor import TrackPreProcessor
from inference.src.tracking.track_processing import TrackPostProcessing
from inference.src.tracking.tracking import Tracker
from inference.src.util.video_io import VideoReader, get_video_properties
from inference.src.util.timing import timeit
from inference.src.visualization.stabilize import (
    MaskedVidStabEstimator,
    STABLE_SMOOTHING_WINDOW,
    Transform,
    stable_processing_max_dim_half,
    vidstab_like_transforms,
)

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
    timeout=600,
    cpu=2.0,
)
# @modal.concurrent(max_inputs=16, target_inputs=12)
def embedding_extraction_and_tracking(
    job_id: str,
    dominant_orientation: int,
    raw_detections: list[dict],
):
    with report_job_failure_on_exception(job_id):
        input_video_path = f'/data/{job_id}_upright.mp4'
        wait_for_volume_reload(input_video_path)

        send_progress(job_id, 'stabilization')

        with timeit(f'{job_id}: Stabilization + crops (single pass)'):
            parsed_transforms, raw_detections_with_crops = _compute_masked_vidstab_transforms_and_crop_detections(
                input_video_path,
                raw_detections,
                mask_margin_px=20,
            )

        send_progress(job_id, 'appearance')
        with timeit(f'{job_id}: Embeddings'):
            detections = EmbeddingExtractor(REID_MODEL_TYPE).run_embedding_pass(raw_detections_with_crops)

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
                        'interpolated': d.interpolated,
                    }
                    for d in t.sorted_detections
                ],
            }
            for t in processed_tracks
        ]

        with timeit(f'{job_id}: Stabilization Optimization'):
            # Compute the transforms which the frontend should use as per frame warps for stabilization
            # stabilized_transforms = optimize_trajectory_world(transforms, properties.width, properties.height)
            smoothing_window = min(int(STABLE_SMOOTHING_WINDOW), props.total_frames - 1)
            stabilized_transforms = vidstab_like_transforms(parsed_transforms, smoothing_window)

        # Frontend convenience: emit an explicit identity transform for frame 0.
        # The rest are prev->curr deltas anchored at their `frame_idx` (e.g. 0->1 has frame_idx==1).
        stabilized_transforms_for_output = [Transform(dx=0.0, dy=0.0, da=0.0, frame_idx=0), *stabilized_transforms]

        # Convert transforms payload into result with time_percent
        stabilization_transforms = [
            {
                # `Transform.frame_idx` is the CURRENT frame index for the prev->curr delta (e.g. 0->1 has frame_idx=1),
                # so we anchor the transform at the current frame timestamp (avoids an off-by-one in visualization).
                'time_percent': clamp_percentage(t.frame_idx / props.total_frames),
                'dx': t.dx,
                'dy': t.dy,
                'da': t.da,
            }
            for t in stabilized_transforms_for_output
        ]

        results = {
            'tracks': tracks,
            'dominant_orientation': dominant_orientation,
            'stabilization_transforms': stabilization_transforms,
        }

        send_complete(job_id, 'succeeded', results)


def _compute_masked_vidstab_transforms_and_crop_detections(
    input_video_path: str,
    raw_detections: list[dict],
    *,
    mask_margin_px: int,
) -> tuple[list[Transform], list[RawDetection]]:
    detections_by_frame: dict[int, list[tuple[BoundingBox, float]]] = defaultdict(list)
    for detection in raw_detections:
        detections_by_frame[int(detection['frame_idx'])].append(
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

    estimator = MaskedVidStabEstimator(processing_max_dim=stable_processing_max_dim_half(input_video_path))
    transforms: list[Transform] = []
    raw_detections_with_crops: list[RawDetection] = []

    with VideoReader(input_video_path) as reader:
        for frame_idx, frame in reader.read_frames():
            frame_idx = int(frame_idx)
            per_frame_dets = detections_by_frame.get(frame_idx, [])

            excluded_bboxes = [([bbox.x1, bbox.y1, bbox.x2, bbox.y2]) for bbox, _ in per_frame_dets]
            transform = estimator.apply(
                frame_idx=frame_idx,
                frame_bgr=frame,
                excluded_bboxes=excluded_bboxes,
                mask_margin_px=int(mask_margin_px),
            )
            if transform is not None:
                transforms.append(transform)

            # Extract crops for embedding in the same pass.
            for bbox, confidence in per_frame_dets:
                x1 = max(0, min(frame.shape[1], int(bbox.x1)))
                y1 = max(0, min(frame.shape[0], int(bbox.y1)))
                x2 = max(0, min(frame.shape[1], int(bbox.x2)))
                y2 = max(0, min(frame.shape[0], int(bbox.y2)))
                if x2 <= x1 or y2 <= y1:
                    continue
                raw_detections_with_crops.append(
                    RawDetection(
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=confidence,
                        frame_idx=frame_idx,
                        crop=frame[y1:y2, x1:x2],
                    )
                )

    return transforms, raw_detections_with_crops
