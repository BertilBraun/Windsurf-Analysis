from __future__ import annotations

import base64
from typing import Optional, Tuple

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .db import SessionLocal
from .models import User


async def get_db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


def parse_basic_auth(request: Request) -> Tuple[str, str]:
    auth = request.headers.get('Authorization')
    if not auth or not auth.lower().startswith('basic '):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Missing basic auth',
            headers={'WWW-Authenticate': f'Basic realm="{settings.BASIC_AUTH_REALM}"'},
        )
    try:
        raw = base64.b64decode(auth.split(' ', 1)[1]).decode('utf-8')
        username, password = raw.split(':', 1)
        return username, password
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid auth header')


pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


async def authenticate_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    username, password = parse_basic_auth(request)

    result = await db.execute(select(User).where(User.username == username))
    user = result.scalars().first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid credentials',
            headers={'WWW-Authenticate': f'Basic realm="{settings.BASIC_AUTH_REALM}"'},
        )
    # Verify bcrypt/argon2 hash stored in DB
    try:
        if not pwd_context.verify(password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Invalid credentials',
                headers={'WWW-Authenticate': f'Basic realm="{settings.BASIC_AUTH_REALM}"'},
            )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid credentials',
            headers={'WWW-Authenticate': f'Basic realm="{settings.BASIC_AUTH_REALM}"'},
        )
    return user
