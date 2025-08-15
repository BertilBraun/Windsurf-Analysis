from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from server.backend.auth import authenticate_user
from server.backend.db import get_db
from server.backend.models import Job, User, Video


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
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Job, Video).where(
            and_(
                Video.original_checksum_sha256 == payload.original_checksum_sha256,
                Job.video_id == Video.id,
                Job.user_id == user.id,
                Job.deleted_at.is_(None),
            )
        )
    )
    existing = result.scalar_one_or_none()
    if existing is None:
        return ChecksumPreflightResponse(exists=False)

    job, video = existing
    return ChecksumPreflightResponse(exists=True, video_id=str(video.id))
