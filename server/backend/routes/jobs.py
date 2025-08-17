from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
import modal
from pydantic import BaseModel
from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from server.backend.auth import authenticate_user
from server.backend.config import Settings
from server.backend.db import get_db
from server.backend.models import Job, JobStatus, Report, ReportType, User, Video
from server.backend.s3 import object_url, s3_client

from server.backend.accessors.job_accessor import (
    get_job_count,
    does_video_exist,
    get_job_by_id_and_user,
    get_job_and_video_by_id_and_user,
    get_job_by_id,
)

router = APIRouter(prefix='/jobs', tags=['jobs'])


class JobCreateUploadResponse(BaseModel):
    job_id: str
    status: Literal['pending', 'running', 'succeeded', 'failed', 'canceled']


class JobSummaryItem(BaseModel):
    id: str
    video_id: str
    model: str
    status: str
    created_at: datetime
    updated_at: datetime


class JobListResponse(BaseModel):
    jobs: list[JobSummaryItem]


class JobDetail(BaseModel):
    id: str
    video_id: str
    model: str
    status: str
    created_at: datetime
    updated_at: datetime
    original_file_path: str
    original_checksum_sha256: str
    tracks: Optional[list[Any]] = None


class ReportRequest(BaseModel):
    message: str
    type: Literal['missed_detection', 'false_association', 'other']


def timestamp_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@router.post('/upload', response_model=JobCreateUploadResponse)
async def jobs_upload(
    file: UploadFile = File(...),
    original_file_path: str = Form(...),
    original_checksum_sha256: str = Form(...),
    yolo_model: str = Form(...),
    reid_model: str = Form(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    if await get_job_count(db, user) >= Settings.MAX_JOBS_PER_USER:
        raise HTTPException(status_code=403, detail={'code': 'quota_exceeded', 'message': 'Job quota exceeded'})

    if await does_video_exist(db, original_checksum_sha256):
        raise HTTPException(status_code=409, detail={'code': 'duplicate_original', 'message': 'Video already exists'})

    ac_key_prefix = f'{Settings.PREFIX_AC_VIDEOS}{original_checksum_sha256}/ac/'

    content = await file.read()
    ac_checksum = hashlib.sha256(content).hexdigest()

    ext = mimetypes.guess_extension(file.content_type or 'video/mp4') or '.mp4'
    ac_key = f'{ac_key_prefix}{ac_checksum}{ext}'
    # TODO? s3 = s3_client()
    # TODO? s3.put_object(Bucket=Settings.S3_BUCKET, Key=ac_key, Body=content, ContentType=file.content_type or 'video/mp4')

    video = Video(
        original_checksum_sha256=original_checksum_sha256,
        original_file_path=original_file_path,
        ac_checksum_sha256=ac_checksum,
        size_bytes=len(content),
        mime_type=file.content_type or 'video/mp4',
        original_name=file.filename,
        ac_storage_url='N/A',  # TODO? object_url(ac_key),
    )
    db.add(video)
    await db.flush()  # Flush to get the video id

    job = Job(user_id=user.id, video_id=video.id, model=f'{yolo_model}-{reid_model}', status=JobStatus.pending)
    db.add(job)
    await db.commit()  # Commit to the database and get the job id

    complete_url = (
        f'{Settings.BACKEND_PUBLIC_BASE_URL}/v1/jobs/{job.id}/complete?secret={Settings.BACKEND_WEBHOOK_SECRET}'
    )

    InferenceModel = modal.Cls.from_name('windsurf-analysis', 'InferenceModel')
    InferenceModel().inference.spawn(
        job_id=str(job.id),
        ac_bytes=content,
        yolo_model=yolo_model,
        reid_model=reid_model,
        complete_webhook=complete_url,
    )

    return JobCreateUploadResponse(job_id=str(job.id), status=job.status.value)


@router.get('', response_model=JobListResponse)
async def list_jobs(
    status_filter: Optional[str] = Query(None, alias='status'),
    updated_after: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    query = select(Job).where(and_(Job.user_id == user.id, Job.deleted_at.is_(None)))
    if status_filter:
        query = query.where(Job.status == JobStatus(status_filter))
    if updated_after:
        query = query.where(Job.updated_at > updated_after)
    query = query.order_by(Job.created_at.desc())

    rows = (await db.execute(query)).scalars().all()
    items = [
        JobSummaryItem(
            id=str(r.id),
            video_id=str(r.video_id),
            model=r.model,
            status=r.status.value,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in rows
    ]
    return JobListResponse(jobs=items)


@router.get('/{job_id}', response_model=JobDetail)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(authenticate_user)):
    existing = await get_job_and_video_by_id_and_user(db, job_id, user)
    if existing is None:
        raise HTTPException(status_code=404, detail='Not found')

    job, video = existing

    await db.execute(update(Video).where(Video.id == job.video_id).values(last_accessed_at=timestamp_now()))

    return JobDetail(
        id=str(job.id),
        video_id=str(job.video_id),
        model=job.model,
        status=job.status.value,
        created_at=job.created_at,
        updated_at=job.updated_at,
        tracks=job.tracks,
        original_file_path=video.original_file_path,
        original_checksum_sha256=video.original_checksum_sha256,
    )


@router.delete('/{job_id}')
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(authenticate_user)):
    job = await get_job_by_id_and_user(db, job_id, user)
    if job is None:
        raise HTTPException(status_code=404, detail='Not found')

    job.deleted_at = timestamp_now()
    await db.flush()  # Flush to update the job

    return {'ok': True}


@router.post('/{job_id}/report')
async def report_job(
    job_id: str, payload: ReportRequest, db: AsyncSession = Depends(get_db), user: User = Depends(authenticate_user)
):
    job = await get_job_by_id_and_user(db, job_id, user)
    if job is None:
        raise HTTPException(status_code=404, detail='Not found')

    db.add(Report(job_id=job.id, type=ReportType(payload.type), message=payload.message))

    return {'ok': True}


class JobsCompleteRequest(BaseModel):
    tracks: list[Any]
    status: Literal['succeeded', 'failed']


@router.post('/{job_id}/complete')
async def jobs_complete(job_id: str, payload: JobsCompleteRequest, secret: str, db: AsyncSession = Depends(get_db)):
    if secret != Settings.BACKEND_WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail='invalid secret')

    job = await get_job_by_id(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Not found')

    job.tracks = payload.tracks
    job.status = JobStatus(payload.status)
    job.finished_at = timestamp_now()
    await db.flush()  # Flush to update the job

    return {'ok': True}
