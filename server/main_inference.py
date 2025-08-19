import time
import tempfile
import requests
from pathlib import Path
from typing import Sequence

import modal

from .inference.src.windsurf_video_processor import (
    SurferDetector,
    get_video_properties,
    GreedyPreprocessor,
    DiscreteILPTracker,
    TrackFilteringSmoothingRelabeling,
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


def clamp_percentage(p: float) -> float:
    return max(0, min(round(p, 5), 1))


@app.cls(gpu='L40S', max_containers=2)
@modal.concurrent(max_inputs=16, target_inputs=12)
class InferenceModel:
    @modal.enter()
    def setup(self):
        self.processors: dict[tuple[str, str], SurferDetector] = {}
        self.orientation_fixer = OrientationFixer('/root/weights/orientation_fixer/best.pt')

    @modal.method()
    def inference(self, job_id: str, ac_bytes: bytes, yolo_model: str, reid_model: str, complete_webhook: str):
        with tempfile.TemporaryDirectory() as td:
            local_video = Path(td) / f'{job_id}.mp4'
            with open(local_video, 'wb') as f:
                f.write(ac_bytes)

            fixed_video = self.orientation_fixer.fix_video(str(local_video))

            key = (yolo_model, reid_model)
            processor = self.processors.get(key)
            if processor is None:
                # Initialize and cache processor for this model pair
                print(f'Initializing processor for {yolo_model} and {reid_model}')
                start = time.time()
                processor = SurferDetector(
                    yolo_model_path='/root/weights/yolo_models/' + yolo_model,
                    reid_model_path='/root/weights/reid_models/' + reid_model,
                )
                print(f'Time taken to initialize processor: {time.time() - start} seconds')
                self.processors[key] = processor

            # For this invocation, write outputs to the temp job directory

            props = get_video_properties(fixed_video)

            with timeit(f'{job_id}: Object detection'):
                detections = processor.run_object_detection_on_video(fixed_video)

            with timeit(f'{job_id}: Trackers'):
                trackers: Sequence[Tracker] = [
                    GreedyPreprocessor(),
                    # GreedyTracker(),
                    DiscreteILPTracker(),
                    TrackFilteringSmoothingRelabeling(),
                ]
                processed_tracks = [
                    Track(track_id=i, sorted_detections=[detection]) for i, detection in enumerate(detections)
                ]
                for tracker in trackers:
                    processed_tracks = tracker.track(processed_tracks, props)

            print(f'{job_id}: Found {len(processed_tracks)} tracks')

            # Convert dataclasses to primitive JSON structure
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

        # POST completion webhook
        print(f'POSTing completion webhook to {complete_webhook}')
        res = requests.post(complete_webhook, json={'status': 'succeeded', 'tracks': tracks}, timeout=60)
        print(f'Completion webhook response: {res.status_code} {res.text}')
