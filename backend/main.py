import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='windsurf-analysis-backend', version='0.1.0')

allowed_origins = [
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
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/hello/world')
def hello_world() -> dict:
    return {'message': 'hello world'}


@app.get('/test')
def test() -> dict:
    return {'message': 'test'}


@app.get('/')
def root() -> dict:
    # Helpful for Cloud Run smoke tests / browser checks
    return {'service': app.title, 'status': 'ok'}


if __name__ == '__main__':
    # Local dev convenience (Cloud Run uses the Docker CMD)
    import uvicorn

    port = int(os.environ.get('PORT', '8080'))
    uvicorn.run('main:app', host='0.0.0.0', port=port, reload=True)
