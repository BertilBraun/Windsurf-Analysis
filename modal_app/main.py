from __future__ import annotations

from contextlib import asynccontextmanager

import modal
from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles

from modal_app.backend.config import Settings
from modal_app.backend.db import init_db
from modal_app.backend.routes.jobs import router as jobs_router
from modal_app.backend.routes.users import router as users_router
from modal_app.backend.routes.videos import router as videos_router

image = modal.Image.debian_slim().pip_install_from_requirements('modal_app/requirements.txt')
app = modal.App('windsurf-analysis', image=image)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


@app.function(secrets=[modal.Secret.from_name('backend-secret')])
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    app = FastAPI(title=Settings.APP_NAME, lifespan=lifespan)

    backend_router = APIRouter(prefix='/api/v1')
    backend_router.include_router(videos_router)
    backend_router.include_router(jobs_router)
    backend_router.include_router(users_router)

    app.include_router(backend_router)

    app.mount('/', StaticFiles(directory='modal_app/frontend/dist'), name='frontend')

    return app
