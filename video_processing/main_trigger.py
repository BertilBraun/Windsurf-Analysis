from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import modal

from config import settings

server_root = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version='3.10')
    .add_local_dir(server_root, remote_path='/root', copy=True, ignore=lambda p: '__pycache__' in p.parts)
    .pip_install_from_requirements(str(server_root / 'requirements.txt'))
)

app = modal.App('windsurf-analysis-trigger', image=image)


class StartJobRequest(BaseModel):
    source_gs_uri: str = Field(min_length=5)
    upright_gs_uri: str = Field(min_length=5)
    yolo_model: str = Field(min_length=1)


@app.function(
    secrets=[modal.Secret.from_name('backend-secret')],
    scaledown_window=60 * 5,
    region='eu-west',
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    api = FastAPI(title='windsurf-analysis-modal-trigger', version='0.1.0', docs_url='/docs', redoc_url=None)

    @api.post('/api/v1/internal/jobs/{job_id}/start')
    def start_job(
        job_id: str,
        payload: StartJobRequest,
        x_modal_secret: str | None = Header(default=None, alias='X-Modal-Secret'),
    ) -> dict[str, bool]:
        if not x_modal_secret or x_modal_secret != settings.modal_shared_secret:
            raise HTTPException(status_code=401, detail='Invalid modal shared secret')

        OrientationModel = modal.Cls.from_name('windsurf-analysis', 'OrientationModel')
        OrientationModel().orient_and_enqueue.spawn(
            job_id=str(job_id),
            yolo_model=str(payload.yolo_model),
            source_gs_uri=str(payload.source_gs_uri),
            upright_gs_uri=str(payload.upright_gs_uri),
        )
        return {'ok': True}

    return api

