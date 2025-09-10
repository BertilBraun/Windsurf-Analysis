from __future__ import annotations

import os

import modal
import requests

from server.inference.src.visualization.stabilize import compute_stabilization_transforms, stabilize_video
from .inference.src.util.timing import timeit
from .inference.src.orientation_fixer import OrientationFixer
from .main_inference import image as inference_image, volume as shared_volume


app = modal.App('windsurf-analysis-stabilization', image=inference_image)


@app.function(
    volumes={'/data': shared_volume},
    scaledown_window=10,
    cpu=2.0,
)
@modal.concurrent(max_inputs=16, target_inputs=12)
def stabilize_and_enqueue(job_id: str, yolo_model: str, reid_model: str, complete_webhook: str):
    try:
        shared_volume.reload()

        input_video_path = f'/data/{job_id}.mp4'
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f'Input video not found: {input_video_path}')

        # Work in ephemeral container FS for intermediates; write final stabilized to /data
        orientation_fixed_video_path = f'{job_id}_fixed_orientation.mp4'
        stabilized_video_path = f'/data/{job_id}_stabilized.mp4'

        with timeit(f'{job_id}: Orientation detection'):
            orientation_fixer = OrientationFixer('/root/weights/orientation_fixer/best.pt')
            dominant_orientation = orientation_fixer.fix_video(input_video_path, orientation_fixed_video_path)

        with timeit(f'{job_id}: Stabilization (CPU)'):
            transforms = compute_stabilization_transforms(orientation_fixed_video_path)
            stabilize_video(
                orientation_fixed_video_path,
                stabilized_video_path,
                transforms,
            )

        # Persist stabilized video into shared volume
        shared_volume.commit()

        # Convert transforms to primitive structure for cross-process call
        transforms_payload = [{'dx': t.dx, 'dy': t.dy, 'da': t.da} for t in transforms]

        # Cleanup temporary file
        try:
            os.remove(orientation_fixed_video_path)
        except Exception:
            pass

        # Enqueue GPU inference continuation
        InferenceModel = modal.Cls.from_name('windsurf-analysis', 'InferenceModel')
        InferenceModel().inference_after_stabilization.spawn(
            job_id=job_id,
            yolo_model=yolo_model,
            reid_model=reid_model,
            dominant_orientation=dominant_orientation,
            transforms=transforms_payload,
            complete_webhook=complete_webhook,
        )
    except Exception as e:
        # If something fails here, forward failure directly to backend webhook to unblock job
        try:
            print(f'Error in stabilization: {e}')
            res = requests.post(
                complete_webhook,
                json={'status': 'failed', 'results': None},
                timeout=60,
            )
            print(f'Completion webhook response (stabilization failed): {res.status_code} {res.text}')
        except Exception as inner:
            print(f'Error posting failure webhook: {inner}')
