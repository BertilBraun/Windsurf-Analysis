import time
import tempfile
from pathlib import Path
from typing import Sequence

import modal


inference_root_folder = Path(__file__).parent

# Container image with system deps for OpenCV/torch
image = (
    modal.Image.debian_slim(python_version='3.10')
    .apt_install('ffmpeg', 'libgl1', 'git')
    .add_local_dir(
        inference_root_folder / 'src',
        remote_path='/root/src',
        copy=True,
        ignore=~modal.FilePatternMatcher('**/*.py'),  # include all .py files (inverted ignore)
    )
    .add_local_dir(inference_root_folder / 'src/weights', remote_path='/root/weights', copy=True)
    .pip_install_from_requirements(str(inference_root_folder / 'requirements.txt'))
)

app = modal.App('windsurf-analysis-inference', image=image)


@app.cls(gpu='L40S', max_containers=2)
@modal.concurrent(max_inputs=16, target_inputs=12)
class InferenceModel:
    @modal.enter()
    def setup(self):
        from .src.windsurf_video_processor import SurferDetector

        self.processors: dict[tuple[str, str], SurferDetector] = {}

    @modal.method()
    def inference(self, job_id: str, ac_bytes: bytes, yolo_model: str, reid_model: str, complete_webhook: str):
        import requests
        from .src.windsurf_video_processor import (
            SurferDetector,
            get_video_properties,
            GreedyPreprocessor,
            DiscreteILPTracker,
            TrackFilteringSmoothingRelabeling,
            Track,
            Tracker,
        )
        from .src.util.timing import timeit

        with tempfile.TemporaryDirectory() as td:
            local_video = Path(td) / f'{job_id}.mp4'
            with open(local_video, 'wb') as f:
                f.write(ac_bytes)

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

            props = get_video_properties(local_video)

            with timeit(f'{job_id}: Object detection'):
                detections = processor.run_object_detection_on_video(local_video)

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
            result = {
                'video_properties': {
                    'fps': props.fps,
                    'width': props.width,
                    'height': props.height,
                    'total_frames': props.total_frames,
                },
                'tracks': [
                    {
                        'track_id': t.track_id,
                        'start_frame': t.start_frame,
                        'end_frame': t.end_frame,
                        'start_time': t.start_frame / props.fps,
                        'duration': t.duration_frames / props.fps,
                        'detection_count': len(t.sorted_detections),
                        'detections': [
                            {
                                'frame_idx': d.frame_idx,
                                'bbox': [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                                'confidence': d.confidence,
                            }
                            for d in t.sorted_detections
                        ],
                    }
                    for t in processed_tracks
                ],
            }

        # POST completion webhook
        requests.post(complete_webhook, json={'status': 'succeeded', 'results_json': result}, timeout=60)
