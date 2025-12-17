from __future__ import annotations

import modal

from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from api.uploads import router as uploads_router

# Reuse the existing Modal image + volume patterns.
server_root = Path(__file__).parent

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

    # CORS: mirror backend/main.py so browser uploads work from local dev + prod.
    # Note: This Modal app is deployed separately from the Cloud Run backend, so it needs
    # its own CORSMiddleware configuration.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            # Firebase Hosting (prod)
            'https://gybelock-00.web.app',
            'https://gybelock.de',
            # Local dev
            'http://localhost',
            'http://localhost:3000',
            'http://localhost:5173',
            'http://127.0.0.1',
            'http://127.0.0.1:3000',
            'http://127.0.0.1:5173',
        ],
        allow_origin_regex=r'^http://(\[::1\]|localhost|127\.0\.0\.1)(:\\d+)?$',
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    @api.options('/{path:path}')
    def cors_preflight(path: str) -> Response:
        # Extra-safety: ensure OPTIONS preflight always returns a response that CORSMiddleware can decorate.
        return Response(status_code=204)

    api.include_router(uploads_router, prefix='/api/v1')
    return api
