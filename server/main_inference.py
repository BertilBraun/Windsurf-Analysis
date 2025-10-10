import modal
import contextlib
import requests

from pathlib import Path

from .inference.src.util.timing import timeit
from .inference.src.tracking.detector import ObjectDetector

inference_root_folder = Path(__file__).parent / 'inference'

# Container image with system deps for OpenCV/torch
image = (
    modal.Image.debian_slim(python_version='3.10')
    .apt_install('ffmpeg', 'libgl1', 'git')
    .add_local_dir(inference_root_folder / 'src', remote_path='/root/src', copy=True)
    .add_local_dir(
        inference_root_folder / 'weights/orientation_fixer', remote_path='/root/weights/orientation_fixer', copy=True
    )
    .add_local_dir(
        inference_root_folder / 'weights/yolo_models',
        remote_path='/root/weights/yolo_models',
        copy=True,
        ignore=lambda p: p.name != 'best.pt',
    )
    .pip_install_from_requirements(str(inference_root_folder / 'requirements.txt'))
)

app = modal.App('windsurf-analysis-inference', image=image)
volume = modal.Volume.from_name('windsurf-analysis-volume', create_if_missing=True)


# failure webhook context manager
@contextlib.contextmanager
def failure_webhook(webhook: str):
    try:
        yield
    except Exception as e:
        print(f'Error in failure_webhook: {e}')
        try:
            requests.post(webhook, json={'status': 'failed', 'error': str(e), 'results': None}, timeout=60)
        except Exception as e:
            print(f'Error posting failure webhook: {e}')


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
        self.processors: dict[str, ObjectDetector] = {}

    def _get_processor(self, yolo_model: str) -> ObjectDetector:
        key = yolo_model
        processor = self.processors.get(key)
        if processor is None:
            # Initialize and cache processor for this model pair
            with timeit(f'Initializing processor for {yolo_model}'):
                processor = ObjectDetector(yolo_model_path='/root/weights/yolo_models/' + yolo_model)
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
        with failure_webhook(complete_webhook):
            volume.reload()

            stabilized_video_path = f'/data/{job_id}.mp4'

            processor = self._get_processor(yolo_model)

            with timeit(f'{job_id}: Object detection'):
                raw_detections = processor.run_detection_pass(stabilized_video_path)

            # Enqueue embedding extraction and tracking
            TrackingFn = modal.Function.from_name('windsurf-analysis', 'embedding_extraction_and_tracking')
            TrackingFn.spawn(
                job_id=str(job_id),
                dominant_orientation=dominant_orientation,
                transforms=transforms,
                raw_detections=[
                    {
                        'bbox': {'x1': d.bbox.x1, 'y1': d.bbox.y1, 'x2': d.bbox.x2, 'y2': d.bbox.y2},
                        'confidence': d.confidence,
                        'frame_idx': d.frame_idx,
                    }
                    for d in raw_detections
                ],
                complete_webhook=complete_webhook,
            )
