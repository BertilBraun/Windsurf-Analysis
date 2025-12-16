import os
import logging
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

import firebase_admin
from firebase_admin import auth
from google.cloud import firestore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('backend')

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
    # Allow any localhost port (covers Vite dev server variations)
    allow_origin_regex=r'^http://(\[::1\]|localhost|127\.0\.0\.1)(:\d+)?$',
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.options('/{path:path}')
def cors_preflight(path: str) -> Response:
    # Extra-safety: ensure OPTIONS preflight always returns a response that CORSMiddleware can decorate.
    return Response(status_code=204)


if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.Client(database='(default)')  # NOTE: Ensure this is the same as in the .env.XXX files in the frontend


@dataclass
class User:
    uid: str
    email: str
    email_verified: bool
    name: str | None = None
    picture: str | None = None


def get_current_user(authorization: str | None = Header(default=None)) -> User:
    if not authorization or not authorization.lower().startswith('bearer '):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Missing Authorization: Bearer <Firebase ID token>',
        )
    token = authorization.split(' ', 1)[1].strip()
    try:
        # clock_skew_seconds helps avoid rare issues if server time is slightly off.
        decoded = auth.verify_id_token(token, clock_skew_seconds=60)
        # If the user has an email, require it to be verified (email/password sign-in).
        user = User(
            uid=decoded['uid'],
            email=decoded['email'],
            email_verified=decoded['email_verified'],
            name=decoded.get('name'),
            picture=decoded.get('picture'),
        )
        if not user.email or not user.email_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Please verify your email address before using this service.',
            )
        return user
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid or expired Firebase ID token')


@app.get('/hello/world')
def hello_world() -> dict:
    return {'message': 'hello world'}


@app.get('/test')
def test() -> dict:
    return {'message': 'test'}


@app.get('/whoami')
def whoami(user: User = Depends(get_current_user)) -> dict:
    return {
        'uid': user.uid,
        'email': user.email,
        'email_verified': user.email_verified,
        'name': user.name,
        'picture': user.picture,
    }


@app.post('/firestore/ping')
def firestore_ping(user: User = Depends(get_current_user)) -> dict:
    try:
        ref = db.collection('backendPings').document(user.uid)
        ref.set({'ts': firestore.SERVER_TIMESTAMP}, merge=True)
        snap = ref.get()
        return {'ok': True, 'uid': user.uid, 'doc': snap.to_dict()}
    except Exception as e:
        log.exception('Error in firestore_ping')
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get('/')
def root() -> dict:
    # Helpful for Cloud Run smoke tests / browser checks
    return {'service': app.title, 'status': 'ok'}


if __name__ == '__main__':
    # Local dev convenience (Cloud Run uses the Docker CMD)
    import uvicorn

    port = int(os.environ.get('PORT', '8080'))
    uvicorn.run('main:app', host='0.0.0.0', port=port, reload=True)
