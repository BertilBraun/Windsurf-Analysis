from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from server.backend.auth import authenticate_user
from server.backend.db import get_db
from server.backend.models import User
from server.backend.accessors.job_accessor import get_job_and_video_by_id_and_user


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
    existing = await get_job_and_video_by_id_and_user(db, payload.original_checksum_sha256, user)
    if existing is None:
        return ChecksumPreflightResponse(exists=False)

    _, video = existing
    return ChecksumPreflightResponse(exists=True, video_id=str(video.id))
