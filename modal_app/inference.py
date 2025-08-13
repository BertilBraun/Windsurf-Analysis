import os
import tempfile
from pathlib import Path

import modal
from typing import Any

# Container image with system deps for OpenCV/torch
image = (
    modal.Image.debian_slim()
    .apt_install('ffmpeg', 'libgl1', 'git')
    .add_local_dir('.', remote_path='/root')
    .pip_install_from_requirements('/root/requirements.txt')
)

app = modal.App('windsurf-analysis-inference')


@app.function(image=image, gpu='A10G', timeout=60 * 30)
def run(
    job_id: str,
    ac_storage_url: str,
    model: str,
    complete_webhook: str,
    s3_endpoint_url: str,
    s3_bucket: str,
    s3_region: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
):
    import boto3  # type: ignore
    import requests
    from botocore.client import Config  # type: ignore

    # Ensure project root is on path
    import sys

    sys.path.append('/root/src')
    from windsurf_video_processor import WindsurfingVideoProcessor

    # Download AC from S3-compatible storage to local temp file
    session = boto3.session.Session()
    s3 = session.client(
        's3',
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
        endpoint_url=s3_endpoint_url,
        region_name=s3_region,
        config=Config(s3={'addressing_style': 'virtual'}),
    )

    assert ac_storage_url.startswith('s3://')
    _, _, rest = ac_storage_url.partition('s3://')
    bucket, _, key = rest.partition('/')

    with tempfile.TemporaryDirectory() as td:
        local_video = Path(td) / 'input.mp4'
        with open(local_video, 'wb') as f:
            obj = s3.get_object(Bucket=bucket or s3_bucket, Key=key)
            f.write(obj['Body'].read())

        # Run pipeline
        processor = WindsurfingVideoProcessor(
            draw_annotations=False,
            output_dir=str(Path(td) / 'out'),
            generate_videos=False,
            debug_views=False,
            parallel_workers=1,
            stabilize=False,
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
def invoke(request: modal.functions.web_endpoint.Request):
    body = request.json
    # Proxy parameters into function call; pass S3 creds so the function can download AC
    run_f: Any = run
    run_f.spawn(
        job_id=body['job_id'],
        ac_storage_url=body['ac_storage_url'],
        model=body.get('model', 'yolo-8n'),
        complete_webhook=body['complete_webhook'],
        s3_endpoint_url=os.environ.get('S3_ENDPOINT_URL'),
        s3_bucket=os.environ.get('S3_BUCKET'),
        s3_region=os.environ.get('S3_REGION', 'auto'),
        s3_access_key_id=os.environ.get('S3_ACCESS_KEY_ID'),
        s3_secret_access_key=os.environ.get('S3_SECRET_ACCESS_KEY'),
    )
    return {'ok': True}
