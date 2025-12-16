from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status

import firebase_admin
from firebase_admin import auth


@dataclass(frozen=True)
class User:
    uid: str
    email: str
    email_verified: bool
    name: str | None = None
    picture: str | None = None


if not firebase_admin._apps:
    firebase_admin.initialize_app()


def get_current_user(authorization: str | None = Header(default=None)) -> User:
    if not authorization or not authorization.lower().startswith('bearer '):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Missing Authorization: Bearer <Firebase ID token>',
        )

    token = authorization.split(' ', 1)[1].strip()
    try:
        decoded = auth.verify_id_token(token, clock_skew_seconds=60)
        user = User(
            uid=decoded['uid'],
            email=decoded.get('email') or '',
            email_verified=bool(decoded.get('email_verified')),
            name=decoded.get('name'),
            picture=decoded.get('picture'),
        )
        # For email/password sign-in, require verified email
        if not user.email or not user.email_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='Please verify your email address before using this service.',
            )
        return user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid or expired Firebase ID token')
