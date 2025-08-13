import tempfile
from pathlib import Path

import modal

# Container image with system deps for OpenCV/torch
image = (
    modal.Image.debian_slim()
    .apt_install('ffmpeg', 'libgl1', 'git')
    .add_local_dir('.', remote_path='/root')
    .pip_install_from_requirements('/root/requirements.txt')
)

app = modal.App('windsurf-analysis-inference')


@app.function(image=image, gpu='A10G', timeout=60 * 30)
def run(job_id: str, ac_bytes: bytes, yolo_model: str, reid_model: str, complete_webhook: str):
    import requests

    # Ensure project root is on path
    import sys

    sys.path.append('/root/src')
    from windsurf_video_processor import WindsurfingVideoProcessor

    with tempfile.TemporaryDirectory() as td:
        local_video = Path(td) / f'{job_id}.mp4'
        with open(local_video, 'wb') as f:
            f.write(ac_bytes)

        # Run pipeline
        processor = WindsurfingVideoProcessor(
            draw_annotations=False,
            output_dir=str(Path(td) / 'out'),
            generate_videos=False,
            debug_views=False,
            parallel_workers=1,
            stabilize=False,
            yolo_model_path=yolo_model,
            reid_model_path=reid_model,
        )
        metadata = processor.process_video(local_video)
        processor.finalize()

        # Convert dataclasses to primitive JSON structure
        result = {
            'video_properties': {
                'fps': metadata.video_properties.fps,
                'width': metadata.video_properties.width,
                'height': metadata.video_properties.height,
                'total_frames': metadata.video_properties.total_frames,
            },
            'tracks': [
                {
                    'track_id': t.track_id,
                    'start_frame': t.start_frame,
                    'end_frame': t.end_frame,
                    'start_time': t.start_time,
                    'duration': t.duration,
                    'detection_count': t.detection_count,
                    'detections': [
                        {'frame_idx': d.frame_idx, 'bbox': d.bbox, 'confidence': d.confidence} for d in t.detections
                    ],
                }
                for t in metadata.tracks
            ],
        }

    # POST completion webhook
    requests.post(complete_webhook, json={'status': 'succeeded', 'results_json': result}, timeout=60)


@app.web_endpoint()
def invoke(request):
    # Expect multipart/form-data with fields: job_id, model, complete_webhook, and file under 'file'
    form = request.form
    files = request.files
    if 'file' not in files:
        return {'error': 'missing file'}, 400
    uploaded = files['file']
    ac_bytes = uploaded.read()

    job_id = form.get('job_id')
    yolo_model = form.get('yolo_model')
    reid_model = form.get('reid_model')
    complete_webhook = form.get('complete_webhook')
    if not job_id or not yolo_model or not reid_model or not complete_webhook:
        return {'error': 'missing job_id or yolo_model or reid_model or complete_webhook'}, 400

    run.spawn(
        job_id=job_id,
        ac_bytes=ac_bytes,
        yolo_model=yolo_model,
        reid_model=reid_model,
        complete_webhook=complete_webhook,
    )
    return {'ok': True}
