from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import modal
from fastapi import FastAPI, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from server.backend.config import Settings
from server.backend.database.db import init_db
from server.backend.routes.jobs import router as jobs_router
from server.backend.routes.users import router as users_router
from server.backend.routes.videos import router as videos_router


server_root_folder = Path(__file__).parent

image = (
    modal.Image.debian_slim()
    .add_local_dir(server_root_folder / 'frontend/dist', remote_path='/root/frontend/dist', copy=True)
    .pip_install_from_requirements(str(server_root_folder / 'requirements.txt'))
)

app = modal.App('windsurf-analysis-backend', image=image)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


@app.function(secrets=[modal.Secret.from_name('backend-secret')], scaledown_window=60 * 5)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    api = FastAPI(
        title=Settings.APP_NAME,
        lifespan=lifespan,
        docs_url='/api/docs',
        redoc_url=None,
    )

    # Allow local dev origins to call the API directly (useful when not using a proxy)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
        allow_credentials=False,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    api.include_router(videos_router, prefix='/api/v1')
    api.include_router(jobs_router, prefix='/api/v1')
    api.include_router(users_router, prefix='/api/v1')

    api.mount('/', StaticFiles(directory='/root/frontend/dist', html=True), name='frontend')

    return api
