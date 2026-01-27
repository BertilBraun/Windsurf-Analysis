"""Firebase authentication utilities and FastAPI dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status

import firebase_admin
from firebase_admin import auth


@dataclass(frozen=True)
class User:
    """Firebase user identity and profile information."""
    uid: str
    email: str
    email_verified: bool
    name: str | None = None
    picture: str | None = None


if not firebase_admin._apps:
    firebase_admin.initialize_app()


def get_current_user_without_email_verification(authorization: str | None = Header(default=None)) -> User:
    """
    Verifies a Firebase ID token from the Authorization header.

    Allows access even if the user's email has not been verified.

    Args:
        authorization: The Bearer token from the request header.

    Returns:
        The verified User object.

    Raises:
        HTTPException: 401 if the token is missing, invalid, or expired.
    """
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
        return user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid or expired Firebase ID token')


def get_current_user(authorization: str | None = Header(default=None)) -> User:
    """
    Verifies a Firebase ID token and requires the user's email to be verified.

    Args:
        authorization: The Bearer token from the request header.

    Returns:
        The verified User object.

    Raises:
        HTTPException: 401 if the token is invalid, or 403 if the email is not verified.
    """
    user = get_current_user_without_email_verification(authorization)
    if not user.email or not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail='Please verify your email address before using this service.'
        )
    return user
