import os

from fastapi import FastAPI

app = FastAPI(title='windsurf-analysis-backend', version='0.1.0')


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
