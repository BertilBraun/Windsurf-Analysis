import os
import logging

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes.jobs import router as jobs_router
from routes.internal_jobs import router as internal_jobs_router
from routes.users import router as users_router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('backend')

app = FastAPI(title='windsurf-analysis-backend', version='0.2.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    # Allow any localhost port (covers Vite dev server variations)
    allow_origin_regex=settings.allow_origin_regex,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.options('/{path:path}')
def cors_preflight(path: str) -> Response:
    # Extra-safety: ensure OPTIONS preflight always returns a response that CORSMiddleware can decorate.
    return Response(status_code=204)


app.include_router(jobs_router)
app.include_router(internal_jobs_router)
app.include_router(users_router)


@app.get('/')
def root() -> dict:
    # Helpful for Cloud Run smoke tests / browser checks
    return {'ok': True}


if __name__ == '__main__':
    # Local dev convenience (Cloud Run uses the Docker CMD)
    import uvicorn

    port = int(os.environ.get('PORT', '8080'))
    uvicorn.run('main:app', host='0.0.0.0', port=port, reload=True)
