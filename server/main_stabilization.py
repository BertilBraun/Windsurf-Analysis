from __future__ import annotations
import os

import modal

from .inference.src.util.timing import timeit
from .inference.src.orientation_fixer import OrientationFixer
from .inference.src.visualization.stabilize import compute_stabilization_transforms
from .main_inference import (
    report_job_failure_on_exception,
    image as inference_image,
    send_progress,
    volume as shared_volume,
    wait_for_volume_reload,
)


app = modal.App('windsurf-analysis-stabilization', image=inference_image)


@app.cls(
    secrets=[modal.Secret.from_name('backend-secret')],
    volumes={'/data': shared_volume},
    scaledown_window=10,
    cpu=2.0,
)
# TODO @modal.concurrent(max_inputs=16, target_inputs=12)
class StabilizationModel:
    @modal.enter()
    def setup(self):
        # Cache OrientationFixer once per container
        self.orientation_fixer = OrientationFixer('/root/weights/orientation_fixer/best.pt')

    @modal.method()
    def stabilize_and_enqueue(self, job_id: str, yolo_model: str):
        with report_job_failure_on_exception(job_id):
            video_path = f'/data/{job_id}.mp4'
            wait_for_volume_reload(video_path)

            send_progress(job_id, 'orientation')

            with timeit(f'{job_id}: Orientation detection'):
                print(f'{job_id}: Starting Orientation detection')
                dominant_orientation = self.orientation_fixer.detect_orientation(video_path)
                print(f'{job_id}: Dominant orientation: {dominant_orientation}')

            oriented_video_path = f'/data/{job_id}_upright.mp4'
            if dominant_orientation != 0:
                self.orientation_fixer.apply_rotation(video_path, oriented_video_path, dominant_orientation)
                os.remove(video_path)
            else:
                os.rename(video_path, oriented_video_path)

            # Persist upright video into shared volume
            shared_volume.commit()

            send_progress(job_id, 'stabilization')

            with timeit(f'{job_id}: Stabilization'):
                print(f'{job_id}: Starting Stabilization')
                transforms = compute_stabilization_transforms(oriented_video_path)

            # Convert transforms to primitive structure for cross-process call
            transforms_payload = [t._asdict() for t in transforms]

            # Enqueue GPU inference continuation
            InferenceModel = modal.Cls.from_name('windsurf-analysis', 'InferenceModel')
            InferenceModel().inference_after_stabilization.spawn(
                job_id=job_id,
                yolo_model=yolo_model,
                dominant_orientation=dominant_orientation,
                transforms=transforms_payload,
            )
