from __future__ import annotations

import base64

from fastapi import Depends, HTTPException, Request, status
from passlib.context import CryptContext

from server.backend.database.db import get_db
from server.backend.database.accessor import DatabaseAccessor
from server.backend.models import User


def parse_basic_auth(request: Request) -> tuple[str, str]:
    auth = request.headers.get('Authorization')
    if not auth or not auth.lower().startswith('basic '):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Missing basic auth')
    try:
        raw = base64.b64decode(auth.split(' ', 1)[1]).decode('utf-8')
        email, password = raw.split(':', 1)
        return email, password
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid auth header')


pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


async def authenticate_user(request: Request, db: DatabaseAccessor = Depends(get_db)) -> User:
    email, password = parse_basic_auth(request)

    user = await db.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')

    if not pwd_context.verify(password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')

    # TODO: Update user last_active_at?

    return user
