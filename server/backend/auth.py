from __future__ import annotations

import base64
import bcrypt

from fastapi import Depends, HTTPException, Request, status

from server.backend.database.db import get_db
from server.backend.database.accessor import DatabaseAccessor, timestamp_now
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


async def authenticate_user(request: Request, db: DatabaseAccessor = Depends(get_db)) -> User:
    email, password = parse_basic_auth(request)

    user = await db.get_user_by_email(email)
    if not user:
        print(f'User not found: {email}')
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')

    if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
        print(f'Invalid password for user: {email}')
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')

    user.last_active_at = timestamp_now()
    await db.flush()

    return user
