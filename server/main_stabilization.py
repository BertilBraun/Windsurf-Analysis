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


@app.function(
    volumes={'/data': shared_volume},
    scaledown_window=10,
    cpu=2.0,
)
@modal.concurrent(max_inputs=16, target_inputs=12)
def stabilize_and_enqueue(job_id: str, yolo_model: str):
    with report_job_failure_on_exception(job_id):
        shared_volume.reload()

        input_video_path = f'/data/{job_id}.mp4'
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f'Input video not found: {input_video_path}')

        # Work in ephemeral container FS for intermediates; write final stabilized to /data
        orientation_fixed_video_path = f'{job_id}_fixed_orientation.mp4'

        with timeit(f'{job_id}: Orientation detection'):
            print(f'{job_id}: Starting Orientation detection')
            orientation_fixer = OrientationFixer('/root/weights/orientation_fixer/best.pt')
            dominant_orientation = orientation_fixer.fix_video(input_video_path, orientation_fixed_video_path)

        send_progress(job_id, 'stabilization')

        with timeit(f'{job_id}: Stabilization'):
            print(f'{job_id}: Starting Stabilization')
            transforms = compute_stabilization_transforms_gmc(orientation_fixed_video_path)

        # Cleanup temporary file and move the stabilized video to the original video path
        os.remove(input_video_path)
        # copy to original video path
        shutil.copy(orientation_fixed_video_path, input_video_path)
        os.remove(orientation_fixed_video_path)

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
