from typing import Literal
import modal
import contextlib
import requests

from pathlib import Path

from server.backend.config import Settings

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
def report_job_failure_on_exception(job_id: str):
    try:
        yield
    except Exception as e:
        print(f'Error in report_job_failure_on_exception: {e}')
        send_complete(job_id, 'failed', None)
        raise e


def send_complete(job_id: str, status: Literal['succeeded', 'failed'], results: dict | None):
    try:
        requests.post(
            f'{Settings.BACKEND_PUBLIC_BASE_URL}/v1/jobs/{job_id}/complete',
            json={'status': status, 'results': results, 'secret': Settings.BACKEND_WEBHOOK_SECRET},
            timeout=60,
        )
    except Exception as e:
        print(f'Error posting complete webhook: {e}')


def send_progress(job_id: str, status: Literal['orientation', 'stabilization', 'detection', 'appearance', 'tracking']):
    try:
        requests.post(
            f'{Settings.BACKEND_PUBLIC_BASE_URL}/v1/jobs/{job_id}/update_progress',
            json={'status': status, 'secret': Settings.BACKEND_WEBHOOK_SECRET},
            timeout=60,
        )
    except Exception as e:
        print(f'Error posting progress webhook: {e}')


@app.cls(
    gpu='T4',
    max_containers=2,
    scaledown_window=5,  # Scaledown window is 5 seconds
    volumes={'/data': volume.read_only()},
    secrets=[modal.Secret.from_name('backend-secret')],
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
    ):
        with report_job_failure_on_exception(job_id):
            volume.reload()

            video_path = f'/data/{job_id}.mp4'

            processor = self._get_processor(yolo_model)
            send_progress(job_id, 'detection')

            with timeit(f'{job_id}: Object detection'):
                raw_detections = processor.run_detection_pass(video_path)

            # Enqueue embedding extraction and tracking
            TrackingFn = modal.Function.from_name('windsurf-analysis', 'embedding_extraction_and_tracking')
            TrackingFn.spawn(
                job_id=str(job_id),
                dominant_orientation=dominant_orientation,
                transforms=transforms,
                raw_detections=[
                    {
                        'bbox': [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                        'confidence': d.confidence,
                        'frame_idx': d.frame_idx,
                    }
                    for d in raw_detections
                ],
            )
