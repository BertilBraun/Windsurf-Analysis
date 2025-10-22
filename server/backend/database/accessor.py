from datetime import datetime, timezone
from typing import Optional, Sequence
import uuid
from sqlalchemy import and_, func, select, update, desc
from sqlalchemy.ext.asyncio import AsyncSession

from server.backend.models import Job, User, UserJob


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

    async def get_total_processed_job_count(self, user: User) -> int:
        # Count all jobs that the user has Processed
        res = await self.db.execute(select(func.count()).select_from(UserJob).where(and_(UserJob.user_id == user.id)))
        return res.scalar_one()

    async def get_job_by_id(self, job_id: str) -> Job | None:
        res = await self.db.execute(select(Job).where(Job.id == uuid.UUID(job_id)))
        return res.scalar_one_or_none()

    async def get_job_by_id_and_user(self, job_id: str, user: User) -> Job | None:
        # Return the job only if the user has an active association
        res = await self.db.execute(
            select(Job)
            .join(UserJob, UserJob.job_id == Job.id)
            .where(
                and_(
                    Job.id == uuid.UUID(job_id),
                    UserJob.user_id == user.id,
                    UserJob.deleted_at.is_(None),
                )
            )
        )
        return res.scalar_one_or_none()

    async def get_user_job_by_job_id_and_user(self, job_id: str, user: User) -> UserJob | None:
        res = await self.db.execute(
            select(UserJob).where(
                and_(
                    UserJob.job_id == uuid.UUID(job_id),
                    UserJob.user_id == user.id,
                    UserJob.deleted_at.is_(None),
                )
            )
        )
        return res.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> User | None:
        res = await self.db.execute(select(User).where(User.email == email).limit(1))
        return res.scalar_one_or_none()

    async def get_jobs_by_user(
        self, user: User, status_filter: Optional[str] = None, updated_after: Optional[datetime] = None
    ) -> Sequence[Job]:
        query = (
            select(Job)
            .join(UserJob, UserJob.job_id == Job.id)
            .where(and_(UserJob.user_id == user.id, UserJob.deleted_at.is_(None)))
        )
        if status_filter:
            query = query.where(Job.status == status_filter)
        if updated_after:
            query = query.where(Job.updated_at > updated_after)
        res = await self.db.execute(query)
        return [row[0] for row in res.all()]

    async def update_job_last_accessed_at(self, job_id: uuid.UUID):
        await self.db.execute(update(Job).where(Job.id == job_id).values(last_accessed_at=timestamp_now()))

    async def get_job_by_original_checksum(self, original_checksum_sha256: str) -> Job | None:
        res = await self.db.execute(
            select(Job).where(Job.original_checksum_sha256 == original_checksum_sha256).limit(1)
        )
        return res.scalar_one_or_none()

    async def ensure_user_job(self, user: User, job: Job) -> UserJob:
        existing = await self.db.execute(
            select(UserJob).where(and_(UserJob.user_id == user.id, UserJob.job_id == job.id))
        )
        row = existing.scalar_one_or_none()
        if row is not None:
            if row.deleted_at is not None:
                row.deleted_at = None
                await self.flush()
            return row
        assoc = UserJob(user_id=user.id, job_id=job.id)
        self.db.add(assoc)
        await self.flush()
        return assoc
