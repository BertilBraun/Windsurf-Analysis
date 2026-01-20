import os
from typing import Literal
import modal
import contextlib
import requests
import tempfile

from pathlib import Path

from config import settings
from gcs_io import download_gs_uri

from inference.src.util.timing import timeit
from inference.src.tracking.detector import ObjectDetector

inference_root_folder = Path(__file__).parent / 'inference'


# Container image with system deps for OpenCV/torch
def ignore_files(p: Path) -> bool:
    # if is .py file, keep it
    if p.name.endswith('.py'):
        return False

    # if is best.pt file, keep it
    if p.name == 'best.pt':
        return False

    return True  # otherwise, ignore it


image = (
    modal.Image.debian_slim(python_version='3.10')
    .apt_install('ffmpeg', 'libgl1', 'git')
    .add_local_dir(
        Path(__file__).parent,
        remote_path='/root',
        copy=True,
        ignore=ignore_files,
    )
    .pip_install_from_requirements(str(inference_root_folder / 'requirements.txt'))
)

app = modal.App('windsurf-analysis-inference', image=image)


# failure webhook context manager
@contextlib.contextmanager
def report_job_failure_on_exception(job_id: str):
    try:
        yield
    except Exception as e:
        print(f'Error in report_job_failure_on_exception: {e}')
        send_complete(job_id, 'failed', None, error_message=str(e))
        raise e


def send_complete(
    job_id: str,
    status: Literal['succeeded', 'failed'],
    results: dict | None,
    *,
    error_message: str | None = None,
):
    try:
        requests.post(
            f'{settings.cloud_run_base_url}/internal/jobs/{job_id}/results',
            json={
                'status': status,
                'results': results,
                'error_message': error_message,
            },
            headers={'X-Modal-Secret': settings.modal_shared_secret},
            timeout=60,
        ).raise_for_status()
    except Exception as e:
        print(f'Error posting complete webhook: {e}')


def send_progress(
    job_id: str,
    status: Literal['orientation', 'stabilization', 'detection', 'appearance', 'tracking'],
):
    try:
        requests.post(
            # Cloud Run only accepts internal job updates under /internal/jobs/...
            f'{settings.cloud_run_base_url}/internal/jobs/{job_id}/status',
            json={'status': status},
            headers={'X-Modal-Secret': settings.modal_shared_secret},
            timeout=60,
        ).raise_for_status()
    except Exception as e:
        print(f'Error posting progress webhook: {e}')


@app.cls(
    gpu='T4',
    max_containers=2,
    scaledown_window=5,  # Scaledown window is 5 seconds
    timeout=600,  # 10 minutes
    secrets=[modal.Secret.from_name('backend-secret')],
)
@modal.concurrent(max_inputs=4, target_inputs=4)
class InferenceModel:
    @modal.enter()
    def setup(self):
        self.processors: dict[str, ObjectDetector] = {}

    def _get_processor(self, yolo_model: str) -> ObjectDetector:
        if yolo_model not in self.processors:
            # Initialize and cache processor for this model pair
            with timeit(f'Initializing processor for {yolo_model}'):
                yolo_model_path = '/root/inference/weights/yolo_models/' + yolo_model
                self.processors[yolo_model] = ObjectDetector(yolo_model_path)
        return self.processors[yolo_model]

    @modal.method()
    def inference_after_orientation(
        self,
        job_id: str,
        yolo_model: str,
        dominant_orientation: int,
        upright_gs_uri: str,
    ):
        with report_job_failure_on_exception(job_id):
            tmpdir = tempfile.gettempdir()
            video_path = os.path.join(tmpdir, f'{job_id}_upright.mp4')
            download_gs_uri(upright_gs_uri, dest_path=video_path)

            processor = self._get_processor(yolo_model)
            send_progress(job_id, 'detection')

            with timeit(f'{job_id}: Object detection'):
                raw_detections = processor.run_detection_pass(video_path)

            # Enqueue embedding extraction and tracking
            TrackingFn = modal.Function.from_name('windsurf-analysis', 'embedding_extraction_and_tracking')
            TrackingFn.spawn(
                job_id=str(job_id),
                dominant_orientation=dominant_orientation,
                upright_gs_uri=upright_gs_uri,
                raw_detections=[
                    {
                        'bbox': [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                        'confidence': d.confidence,
                        'frame_idx': d.frame_idx,
                        'boom': [d.boom.point.x, d.boom.point.y, float(d.boom.conf)],
                        'mast_tip': [d.mast_tip.point.x, d.mast_tip.point.y, float(d.mast_tip.conf)],
                    }
                    for d in raw_detections
                ],
            )
