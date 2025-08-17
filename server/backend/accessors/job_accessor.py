from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Job, Video, User
import uuid


async def get_job_count(db: AsyncSession, user: User) -> int:
    res = await db.execute(select(func.count()).select_from(Job).where(Job.user_id == user.id))
    return res.scalar_one()


async def does_video_exist(db: AsyncSession, original_checksum_sha256: str) -> bool:
    res = await db.execute(select(Video).where(Video.original_checksum_sha256 == original_checksum_sha256))
    return res.scalar_one_or_none() is not None


async def get_job_by_id(db: AsyncSession, job_id: str) -> Job | None:
    res = await db.execute(select(Job).where(Job.id == uuid.UUID(job_id)))
    return res.scalar_one_or_none()


async def get_job_by_id_and_user(db: AsyncSession, job_id: str, user: User) -> Job | None:
    res = await db.execute(select(Job).where(and_(Job.id == uuid.UUID(job_id), Job.user_id == user.id)))
    return res.scalar_one_or_none()


async def get_job_and_video_by_id_and_user(db: AsyncSession, job_id: str, user: User) -> tuple[Job, Video] | None:
    res = await db.execute(
        select(Job, Video)
        .join(Video, Job.video_id == Video.id)
        .where(
            and_(
                Job.id == uuid.UUID(job_id),
                Job.user_id == user.id,
                Job.deleted_at.is_(None),
            )
        )
    )
    result = res.one_or_none()
    return result.t if result else None
