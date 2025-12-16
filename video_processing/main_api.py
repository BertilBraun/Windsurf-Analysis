from __future__ import annotations

import modal

from pathlib import Path
from fastapi import FastAPI

from api.uploads import router as uploads_router

# Reuse the existing Modal image + volume patterns.
root = Path(__file__).resolve().parents[1]
server_root = root / 'video_processing'

image = (
    modal.Image.debian_slim()
    .add_local_dir(server_root, remote_path='/root', copy=True, ignore=lambda p: '__pycache__' in p.parts)
    .pip_install_from_requirements(str(server_root / 'requirements.txt'))
)

app = modal.App('windsurf-analysis-upload-only', image=image)
volume = modal.Volume.from_name('windsurf-analysis-volume', create_if_missing=True)


@app.function(
    secrets=[modal.Secret.from_name('backend-secret')],
    scaledown_window=60 * 5,
    region='eu-west',
    volumes={'/data': volume},
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    api = FastAPI(title='windsurf-analysis-modal-upload', version='0.1.0', docs_url='/docs', redoc_url=None)
    # Keep browser-facing endpoints compatible with existing UI (API_BASE ends with /api/v1).
    api.include_router(uploads_router, prefix='/api/v1')
    return api
