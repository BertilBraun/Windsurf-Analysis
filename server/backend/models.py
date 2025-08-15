from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Enum, ForeignKey, Index, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import CITEXT, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class JobStatus(enum.Enum):
    pending = 'pending'
    running = 'running'
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
    last_active_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    jobs: Mapped[list[Job]] = relationship('Job', back_populates='user')


class Video(Base):
    __tablename__ = 'videos'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    original_checksum_sha256: Mapped[str] = mapped_column(unique=True, nullable=False)
    original_file_path: Mapped[str] = mapped_column(nullable=False)
    ac_checksum_sha256: Mapped[str] = mapped_column(nullable=False)
    size_bytes: Mapped[int] = mapped_column(nullable=False)
    mime_type: Mapped[str] = mapped_column(nullable=False)
    original_name: Mapped[str | None] = mapped_column(nullable=True)
    ac_storage_url: Mapped[str] = mapped_column(nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_accessed_at: Mapped[datetime] = mapped_column(server_default=func.now())

    jobs: Mapped[list[Job]] = relationship('Job', back_populates='video')

    __table_args__ = (
        Index('idx_videos_original', 'original_checksum_sha256'),
        Index('idx_videos_last_accessed', 'last_accessed_at'),
    )


class Job(Base):
    __tablename__ = 'jobs'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('videos.id', ondelete='RESTRICT'), nullable=False)
    model: Mapped[str] = mapped_column(nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.pending, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error_message: Mapped[str | None] = mapped_column(nullable=True)
    results_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(nullable=True)

    user: Mapped[User] = relationship('User', back_populates='jobs')
    video: Mapped[Video] = relationship('Video', back_populates='jobs')
    reports: Mapped[list[Report]] = relationship('Report', back_populates='job')

    __table_args__ = (
        UniqueConstraint('user_id', 'video_id', 'model', name='uq_user_video_model'),
        Index('idx_jobs_user_created', 'user_id', text('created_at DESC')),
    )


class Report(Base):
    __tablename__ = 'reports'

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False)
    type: Mapped[ReportType] = mapped_column(Enum(ReportType), nullable=False)
    message: Mapped[str] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    job: Mapped[Job] = relationship('Job', back_populates='reports')
