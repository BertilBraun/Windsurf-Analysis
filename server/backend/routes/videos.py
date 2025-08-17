from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from server.backend.auth import authenticate_user
from server.backend.database.db import get_db
from server.backend.models import User
from server.backend.database.accessor import DatabaseAccessor


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
    db: DatabaseAccessor = Depends(get_db),
):
    existing = await db.get_job_and_video_by_checksum_and_user(payload.original_checksum_sha256, user)
    if existing is None:
        return ChecksumPreflightResponse(exists=False)

    _, video = existing
    return ChecksumPreflightResponse(exists=True, video_id=str(video.id))
