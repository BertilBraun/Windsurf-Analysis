from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from server.backend.auth import authenticate_user
from server.backend.config import Settings
from server.backend.database.accessor import DatabaseAccessor, timestamp_now
from server.backend.database.db import get_db
from server.backend.models import Job, JobStatus, Report, ReportType, User, Video

# from server.backend.s3 import object_url, s3_client


router = APIRouter(prefix='/jobs', tags=['jobs'])

job_status = Literal['pending', 'running', 'succeeded', 'failed', 'canceled']


class JobCreateResponse(BaseModel):
    job_id: str
    status: job_status


class JobSummaryItem(BaseModel):
    id: str
    video_id: str
    status: job_status
    created_at: datetime
    updated_at: datetime
    original_checksum_sha256: str
    dominant_orientation: int


class JobListResponse(BaseModel):
    jobs: list[JobSummaryItem]


class JobDetail(JobSummaryItem):
    tracks: list[Any]
    stabilization_transforms: list[Any]


class ReportRequest(BaseModel):
    message: str
    type: Literal['missed_detection', 'false_association', 'other']


class JobCreateRequest(BaseModel):
    original_checksum_sha256: str


@router.post('', response_model=JobCreateResponse)
async def create_job(
    payload: JobCreateRequest,
    db: DatabaseAccessor = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    if await db.get_job_count(user) >= user.max_jobs_per_user:
        raise HTTPException(status_code=403, detail={'code': 'quota_exceeded', 'message': 'Job quota exceeded'})

    if await db.does_video_exist(payload.original_checksum_sha256):
        raise HTTPException(status_code=409, detail={'code': 'duplicate_original', 'message': 'Video already exists'})

    # Create placeholder video record
    video = Video(
        original_checksum_sha256=payload.original_checksum_sha256,
        ac_checksum_sha256='PENDING',
        size_bytes=-1,
        mime_type='video/mp4',
        ac_storage_url='N/A',
    )
    await db.add(video)

    job = Job(user_id=user.id, video_id=video.id, status=JobStatus.pending)
    await db.add(job)
    await db.flush()

    return JobCreateResponse(job_id=str(job.id), status=job.status.value)


@router.get('', response_model=JobListResponse)
async def list_jobs(
    status_filter: Optional[str] = Query(None, alias='status'),
    updated_after: Optional[datetime] = Query(None),
    db: DatabaseAccessor = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    rows = await db.get_jobs_by_user(user, status_filter, updated_after)
    items = [
        JobSummaryItem(
            id=str(job.id),
            video_id=str(job.video_id),
            status=job.status.value,
            created_at=job.created_at,
            updated_at=job.updated_at,
            original_checksum_sha256=video.original_checksum_sha256,
            dominant_orientation=job.results['dominant_orientation'] if job.results else 0,
        )
        for job, video in rows
    ]
    return JobListResponse(jobs=items)


@router.get('/{job_id}', response_model=JobDetail)
async def get_job(job_id: str, db: DatabaseAccessor = Depends(get_db), user: User = Depends(authenticate_user)):
    existing = await db.get_job_and_video_by_id_and_user(job_id, user)
    if existing is None:
        raise HTTPException(status_code=404, detail='Not found')

    job, video = existing

    await db.update_video_last_accessed_at(job.video_id)
    await db.flush()

    if job.status != JobStatus.succeeded or job.results is None:
        raise HTTPException(status_code=405, detail='Results not found')

    return JobDetail(
        id=str(job.id),
        video_id=str(job.video_id),
        status=job.status.value,
        created_at=job.created_at,
        updated_at=job.updated_at,
        tracks=job.results['tracks'],
        stabilization_transforms=job.results['stabilization_transforms'],
        original_checksum_sha256=video.original_checksum_sha256,
        dominant_orientation=job.results['dominant_orientation'],
    )


@router.delete('/{job_id}')
async def delete_job(job_id: str, db: DatabaseAccessor = Depends(get_db), user: User = Depends(authenticate_user)):
    job = await db.get_job_by_id_and_user(job_id, user)
    if job is None:
        raise HTTPException(status_code=404, detail='Not found')

    job.deleted_at = timestamp_now()
    await db.flush()

    return {'ok': True}


@router.post('/{job_id}/report')
async def report_job(
    job_id: str,
    payload: ReportRequest,
    db: DatabaseAccessor = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    job = await db.get_job_by_id_and_user(job_id, user)
    if job is None:
        raise HTTPException(status_code=404, detail='Not found')

    await db.add(Report(job_id=job.id, type=ReportType(payload.type), message=payload.message))
    await db.flush()

    return {'ok': True}


class JobsCompleteResults(BaseModel):
    tracks: list[Any]
    dominant_orientation: int
    stabilization_transforms: list[Any]


class JobsCompleteRequest(BaseModel):
    status: Literal['succeeded', 'failed']
    results: JobsCompleteResults | None


@router.post('/{job_id}/complete')
async def jobs_complete(
    job_id: str,
    payload: JobsCompleteRequest,
    secret: str,
    db: DatabaseAccessor = Depends(get_db),
):
    if secret != Settings.BACKEND_WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail='invalid secret')

    job = await db.get_job_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Not found')

    # Optionally delete the video files
    try:
        os.remove(f'/data/{job_id}.mp4')
    except Exception as e:
        print(f'Error deleting video file: {e}')

    if payload.status != 'succeeded' or payload.results is None:
        job.error_message = 'Failed to complete job'
        job.status = JobStatus.failed
        job.finished_at = timestamp_now()
        await db.flush()
        return {'ok': True}

    job.results = {
        'tracks': payload.results.tracks,
        'dominant_orientation': payload.results.dominant_orientation,
        'stabilization_transforms': payload.results.stabilization_transforms,
    }
    job.status = JobStatus.succeeded
    job.finished_at = timestamp_now()
    await db.flush()

    return {'ok': True}
