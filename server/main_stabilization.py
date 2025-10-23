from __future__ import annotations

import os
import shutil

import modal

from .inference.src.util.timing import timeit
from .inference.src.orientation_fixer import OrientationFixer
from .inference.src.visualization.stabilize import compute_stabilization_transforms_gmc
from .main_inference import (
    report_job_failure_on_exception,
    image as inference_image,
    send_progress,
    volume as shared_volume,
)


app = modal.App('windsurf-analysis-stabilization', image=inference_image)


@app.cls(
    secrets=[modal.Secret.from_name('backend-secret')],
    volumes={'/data': shared_volume},
    scaledown_window=10,
    cpu=2.0,
)
@modal.concurrent(max_inputs=16, target_inputs=12)
class StabilizationModel:
    @modal.enter()
    def setup(self):
        # Cache OrientationFixer once per container
        self.orientation_fixer = OrientationFixer('/root/weights/orientation_fixer/best.pt')

    @modal.method()
    def stabilize_and_enqueue(self, job_id: str, yolo_model: str):
        with report_job_failure_on_exception(job_id):
            shared_volume.reload()

            shared_video_path = f'/data/{job_id}.mp4'
            if not os.path.exists(shared_video_path):
                raise FileNotFoundError(f'Input video not found: {shared_video_path}')

            with timeit(f'{job_id}: Orientation detection'):
                print(f'{job_id}: Starting Orientation detection')
                input_video_path = 'tmp.mp4'
                shutil.copy(shared_video_path, input_video_path)
                # Write the fixed video back to the shared volume
                dominant_orientation = self.orientation_fixer.fix_video(input_video_path, shared_video_path)
                os.remove(input_video_path)

            send_progress(job_id, 'stabilization')

            with timeit(f'{job_id}: Stabilization'):
                print(f'{job_id}: Starting Stabilization')
                transforms = compute_stabilization_transforms_gmc(shared_video_path)

            # Persist stabilized video into shared volume
            shared_volume.commit()

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
