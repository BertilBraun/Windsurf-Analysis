from __future__ import annotations

import os
import shutil

import modal

from server.inference.src.visualization.stabilize import compute_stabilization_transforms, stabilize_video
from .inference.src.util.timing import timeit
from .inference.src.orientation_fixer import OrientationFixer
from .main_inference import failure_webhook, image as inference_image, volume as shared_volume


app = modal.App('windsurf-analysis-stabilization', image=inference_image)


@app.function(
    volumes={'/data': shared_volume},
    scaledown_window=10,
    cpu=2.0,
)
@modal.concurrent(max_inputs=16, target_inputs=12)
def stabilize_and_enqueue(job_id: str, yolo_model: str, complete_webhook: str):
    with failure_webhook(complete_webhook):
        shared_volume.reload()

        input_video_path = f'/data/{job_id}.mp4'
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f'Input video not found: {input_video_path}')

        # Work in ephemeral container FS for intermediates; write final stabilized to /data
        orientation_fixed_video_path = f'{job_id}_fixed_orientation.mp4'
        stabilized_video_path = f'{job_id}_stabilized.mp4'

        with timeit(f'{job_id}: Orientation detection'):
            print(f'{job_id}: Starting Orientation detection')
            orientation_fixer = OrientationFixer('/root/weights/orientation_fixer/best.pt')
            dominant_orientation = orientation_fixer.fix_video(input_video_path, orientation_fixed_video_path)

        with timeit(f'{job_id}: Stabilization (CPU)'):
            print(f'{job_id}: Starting Stabilization')
            transforms = compute_stabilization_transforms(orientation_fixed_video_path)
            stabilize_video(
                orientation_fixed_video_path,
                stabilized_video_path,
                transforms,
            )

        # Cleanup temporary file and move the stabilized video to the original video path
        os.remove(input_video_path)
        # copy to original video path
        shutil.copy(stabilized_video_path, input_video_path)
        os.remove(stabilized_video_path)
        os.remove(orientation_fixed_video_path)

        # Persist stabilized video into shared volume
        shared_volume.commit()

        # Convert transforms to primitive structure for cross-process call
        transforms_payload = [{'dx': t.dx, 'dy': t.dy, 'da': t.da} for t in transforms]

        # Enqueue GPU inference continuation
        InferenceModel = modal.Cls.from_name('windsurf-analysis', 'InferenceModel')
        InferenceModel().inference_after_stabilization.spawn(
            job_id=job_id,
            yolo_model=yolo_model,
            dominant_orientation=dominant_orientation,
            transforms=transforms_payload,
            complete_webhook=complete_webhook,
        )
