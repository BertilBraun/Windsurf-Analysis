from datetime import datetime, timezone
from typing import Optional, Sequence
import uuid
from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from server.backend.models import Job, Video, User


def timestamp_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DatabaseAccessor:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def commit(self):
        await self.db.commit()

    async def flush(self):
        await self.db.flush()

    async def add(self, obj: object):
        self.db.add(obj)
        await self.flush()

    async def get_job_count(self, user: User) -> int:
        res = await self.db.execute(select(func.count()).select_from(Job).where(Job.user_id == user.id))
        return res.scalar_one()

    async def does_video_exist(self, original_checksum_sha256: str) -> bool:
        res = await self.db.execute(select(Video).where(Video.original_checksum_sha256 == original_checksum_sha256))
        return res.scalar_one_or_none() is not None

    async def get_job_by_id(self, job_id: str) -> Job | None:
        res = await self.db.execute(select(Job).where(Job.id == uuid.UUID(job_id)))
        return res.scalar_one_or_none()

    async def get_job_by_id_and_user(self, job_id: str, user: User) -> Job | None:
        res = await self.db.execute(select(Job).where(and_(Job.id == uuid.UUID(job_id), Job.user_id == user.id)))
        return res.scalar_one_or_none()

    async def get_job_and_video_by_id_and_user(self, job_id: str, user: User) -> tuple[Job, Video] | None:
        res = await self.db.execute(
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

    async def get_job_and_video_by_checksum_and_user(
        self, original_checksum_sha256: str, user: User
    ) -> tuple[Job, Video] | None:
        res = await self.db.execute(
            select(Job, Video)
            .join(Video, Job.video_id == Video.id)
            .where(
                and_(
                    Video.original_checksum_sha256 == original_checksum_sha256,
                    Job.user_id == user.id,
                    Job.deleted_at.is_(None),
                )
            )
        )
        result = res.one_or_none()
        return result.t if result else None

    async def get_user_by_email(self, email: str) -> User | None:
        res = await self.db.execute(select(User).where(User.email == email))
        return res.scalar_one_or_none()

    async def get_jobs_by_user(
        self, user: User, status_filter: Optional[str] = None, updated_after: Optional[datetime] = None
    ) -> Sequence[tuple[Job, Video]]:
        query = select(Job, Video).join(Video, Job.video_id == Video.id).where(Job.user_id == user.id)
        if status_filter:
            query = query.where(Job.status == status_filter)
        if updated_after:
            query = query.where(Job.updated_at > updated_after)
        res = await self.db.execute(query)
        return [row.t for row in res.all()]

    async def update_video_last_accessed_at(self, video_id: uuid.UUID):
        await self.db.execute(update(Video).where(Video.id == video_id).values(last_accessed_at=timestamp_now()))
