from __future__ import annotations

from fastapi import Header, HTTPException, status

from config import settings


def require_modal_secret(x_modal_secret: str | None = Header(default=None, alias=settings.modal_secret_header)) -> None:
    expected = settings.modal_shared_secret
    if not expected:
        # Fail closed if not configured.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Internal auth not configured (MODAL_SHARED_SECRET missing)',
        )
    if not x_modal_secret or x_modal_secret != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='invalid secret')

