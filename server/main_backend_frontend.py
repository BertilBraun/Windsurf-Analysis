from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import modal
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server.backend.config import Settings
from server.backend.db import init_db
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


@app.function(secrets=[modal.Secret.from_name('backend-secret')])
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    api = FastAPI(
        title=Settings.APP_NAME,
        lifespan=lifespan,
        docs_url='/api/docs',
        redoc_url=None,
    )

    api.include_router(videos_router, prefix='/api/v1')
    api.include_router(jobs_router, prefix='/api/v1')
    api.include_router(users_router, prefix='/api/v1')

    api.mount('/', StaticFiles(directory='/root/frontend/dist', html=True), name='frontend')

    return api
