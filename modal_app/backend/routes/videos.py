from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select

from modal_app.backend.auth import authenticate_user
from modal_app.backend.db import session_scope
from modal_app.backend.models import User, Video


router = APIRouter(prefix='/videos', tags=['videos'])


class ChecksumPreflightRequest(BaseModel):
    original_checksum_sha256: str = Field(min_length=64, max_length=64)


class ChecksumPreflightResponse(BaseModel):
    exists: bool
    video_id: Optional[str] = None


@router.post('/checksum', response_model=ChecksumPreflightResponse)
async def videos_checksum(
    payload: ChecksumPreflightRequest,
    user: User = Depends(authenticate_user),
):
    async with session_scope() as db:
        result = await db.execute(
            select(Video).where(Video.original_checksum_sha256 == payload.original_checksum_sha256)
        )
        existing = result.scalars().first()
        if existing:
            return ChecksumPreflightResponse(exists=True, video_id=str(existing.id))
        return ChecksumPreflightResponse(exists=False)
