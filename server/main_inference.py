import requests
from pathlib import Path
from typing import Sequence

import modal

from .inference.src.windsurf_video_processor import (
    SurferDetector,
    get_video_properties,
    Preprocessor,
    DiscreteILPTracker,
    TrackFiltering,
    TrackRelabeling,
    Track,
    Tracker,
)
from .inference.src.util.timing import timeit
from .inference.src.orientation_fixer import OrientationFixer

inference_root_folder = Path(__file__).parent / 'inference'

# Container image with system deps for OpenCV/torch
image = (
    modal.Image.debian_slim(python_version='3.10')
    .apt_install('ffmpeg', 'libgl1', 'git')
    .add_local_dir(inference_root_folder / 'src', remote_path='/root/src', copy=True)
    .add_local_dir(inference_root_folder / 'weights', remote_path='/root/weights', copy=True)
    .pip_install_from_requirements(str(inference_root_folder / 'requirements.txt'))
)

app = modal.App('windsurf-analysis-inference', image=image)
volume = modal.Volume.from_name('windsurf-analysis-volume', create_if_missing=True)


def clamp_percentage(p: float) -> float:
    return max(0, min(p, 1))


@app.cls(
    gpu='T4',
    max_containers=2,
    scaledown_window=10,  # Scaledown window is 10 seconds
    volumes={'/data': volume.read_only()},
)
@modal.concurrent(max_inputs=16, target_inputs=12)
class InferenceModel:
    @modal.enter()
    def setup(self):
        self.processors: dict[str, SurferDetector] = {}
        self.orientation_fixer = OrientationFixer('/root/weights/orientation_fixer/best.pt')

    def _get_processor(self, yolo_model: str) -> SurferDetector:
        key = yolo_model
        processor = self.processors.get(key)
        if processor is None:
            # Initialize and cache processor for this model pair
            with timeit(f'Initializing processor for {yolo_model}'):
                processor = SurferDetector(yolo_model_path='/root/weights/yolo_models/' + yolo_model)
                self.processors[key] = processor
        return processor

    @modal.method()
    def inference_after_stabilization(
        self,
        job_id: str,
        yolo_model: str,
        dominant_orientation: int,
        transforms: list[dict],
        complete_webhook: str,
    ):
        def _post_completion_webhook(status: str, results: dict | None):
            print(f'POSTing completion webhook to {complete_webhook}')
            res = requests.post(
                complete_webhook,
                json={'status': status, 'results': results},
                timeout=60,
            )
            print(f'Completion webhook response: {res.status_code} {res.text}')

        try:
            volume.reload()

            stabilized_video_path = f'/data/{job_id}_stabilized.mp4'

            processor = self._get_processor(yolo_model)

            props = get_video_properties(stabilized_video_path)

            with timeit(f'{job_id}: Object detection'):
                detections = processor.run_object_detection_on_video(stabilized_video_path)

            with timeit(f'{job_id}: Trackers'):
                trackers: Sequence[Tracker] = [
                    Preprocessor(),
                    # GreedyTracker(),
                    DiscreteILPTracker(),
                    TrackFiltering(),
                    # TrackInterpolation(), # Not needed - done in frontend
                    # TrackSmoothing(), # Not needed - done in frontend
                    TrackRelabeling(),
                ]
                processed_tracks = [
                    Track(track_id=i, sorted_detections=[detection]) for i, detection in enumerate(detections)
                ]
                for tracker in trackers:
                    processed_tracks = tracker.track(processed_tracks, props)

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
                    'time_percent': clamp_percentage(i / max(1, len(transforms))),
                    'dx': t['dx'],
                    'dy': t['dy'],
                    'da': t['da'],
                }
                for i, t in enumerate(transforms)
            ]

            results = {
                'tracks': tracks,
                'dominant_orientation': dominant_orientation,
                'stabilization_transforms': stabilization_transforms,
            }

            _post_completion_webhook('succeeded', results)
        except Exception as e:
            print(f'Error in inference_after_stabilization: {e}')
            _post_completion_webhook('failed', None)
