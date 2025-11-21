from __future__ import annotations

import enum
from typing import Any
import uuid
from datetime import datetime

from sqlalchemy import Enum, ForeignKey, Index, func, text
from sqlalchemy.dialects.postgresql import CITEXT, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from server.backend.config import Settings


class Base(DeclarativeBase):
    pass


class JobStatus(enum.Enum):
    pending = 'pending'
    starting = 'starting'
    orientation = 'orientation'
    stabilization = 'stabilization'
    detection = 'detection'
    appearance = 'appearance'
    tracking = 'tracking'
    succeeded = 'succeeded'
    failed = 'failed'
    canceled = 'canceled'


class ReportType(enum.Enum):
    missed_detection = 'missed_detection'
    false_association = 'false_association'
    other = 'other'


class User(Base):
    __tablename__ = 'users'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(CITEXT, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(nullable=False)
    # Per-user quota for maximum concurrent/total jobs allowed
    max_jobs_per_user: Mapped[int] = mapped_column(
        nullable=False,
        default=Settings.MAX_JOBS_PER_USER,
        server_default=text(str(Settings.MAX_JOBS_PER_USER)),
    )
    last_active_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Association to jobs is now via UserJob
    user_jobs: Mapped[list[UserJob]] = relationship('UserJob', back_populates='user')


class Job(Base):
    __tablename__ = 'jobs'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Merged former Video fields directly into Job
    original_checksum_sha256: Mapped[str] = mapped_column(unique=True, nullable=False)
    ac_checksum_sha256: Mapped[str] = mapped_column(nullable=False)
    size_bytes: Mapped[int] = mapped_column(nullable=False)
    mime_type: Mapped[str] = mapped_column(nullable=False)
    ac_storage_url: Mapped[str] = mapped_column(nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_accessed_at: Mapped[datetime] = mapped_column(server_default=func.now())
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.pending, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(nullable=True)
    results: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Associations
    user_jobs: Mapped[list[UserJob]] = relationship('UserJob', back_populates='job')
    reports: Mapped[list[Report]] = relationship('Report', back_populates='job')

    __table_args__ = (
        Index('idx_jobs_original', 'original_checksum_sha256'),
        Index('idx_jobs_last_accessed', 'last_accessed_at'),
        Index('idx_jobs_created', text('created_at DESC')),
    )


class UserJob(Base):
    __tablename__ = 'user_jobs'

    # Composite primary key (user_id, job_id)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'), primary_key=True)
    deleted_at: Mapped[datetime | None] = mapped_column(nullable=True)

    user: Mapped[User] = relationship('User', back_populates='user_jobs')
    job: Mapped[Job] = relationship('Job', back_populates='user_jobs')

    __table_args__ = (Index('idx_user_jobs_user', 'user_id'),)


class Report(Base):
    __tablename__ = 'reports'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False)
    type: Mapped[ReportType] = mapped_column(Enum(ReportType), nullable=False)
    message: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    job: Mapped[Job] = relationship('Job', back_populates='reports')
