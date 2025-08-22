from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
import modal
from pydantic import BaseModel

from server.backend.auth import authenticate_user
from server.backend.config import Settings
from server.backend.database.accessor import DatabaseAccessor, timestamp_now
from server.backend.database.db import get_db
from server.backend.models import Job, JobStatus, Report, ReportType, User, Video

# from server.backend.s3 import object_url, s3_client
from server.inference.src.util.timing import timeit


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
    original_file_path: str
    original_checksum_sha256: str
    dominant_orientation: Optional[int] = None


class JobListResponse(BaseModel):
    jobs: list[JobSummaryItem]


class JobDetail(JobSummaryItem):
    tracks: Optional[list[Any]] = None


class ReportRequest(BaseModel):
    message: str
    type: Literal['missed_detection', 'false_association', 'other']


class JobCreateRequest(BaseModel):
    original_file_path: str
    original_checksum_sha256: str


@router.post('', response_model=JobCreateResponse)
async def create_job(
    payload: JobCreateRequest,
    db: DatabaseAccessor = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    if await db.get_job_count(user) >= Settings.MAX_JOBS_PER_USER:
        raise HTTPException(status_code=403, detail={'code': 'quota_exceeded', 'message': 'Job quota exceeded'})

    if await db.does_video_exist(payload.original_checksum_sha256):
        raise HTTPException(status_code=409, detail={'code': 'duplicate_original', 'message': 'Video already exists'})

    # Create placeholder video record
    video = Video(
        original_checksum_sha256=payload.original_checksum_sha256,
        original_file_path=payload.original_file_path,
        ac_checksum_sha256='PENDING',
        size_bytes=-1,
        mime_type='video/mp4',
        original_name=None,
        ac_storage_url='N/A',
    )
    await db.add(video)

    job = Job(user_id=user.id, video_id=video.id, status=JobStatus.pending)
    await db.add(job)
    await db.commit()

    return JobCreateResponse(job_id=str(job.id), status=job.status.value)


@router.post('/{job_id}/upload')
async def jobs_upload_for_created_job(
    job_id: str,
    file: UploadFile = File(...),
    yolo_model: str = Form(...),
    reid_model: str = Form(...),
    db: DatabaseAccessor = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    existing = await db.get_job_and_video_by_id_and_user(job_id, user)
    if existing is None:
        raise HTTPException(status_code=404, detail='Not found')

    job, video = existing
    if job.status not in (JobStatus.pending,):
        raise HTTPException(status_code=409, detail='Job not in a state that accepts uploads')

    # ac_key_prefix = f"{Settings.PREFIX_AC_VIDEOS}{video.original_checksum_sha256}/ac/"

    content = await file.read()
    ac_checksum = hashlib.sha256(content).hexdigest()

    # ac_key = f"{ac_key_prefix}{ac_checksum}.mp4"
    # TODO? s3 = s3_client()
    # TODO? s3.put_object(Bucket=Settings.S3_BUCKET, Key=ac_key, Body=content, ContentType=file.content_type or 'video/mp4')

    # Update video with actual uploaded info
    video.ac_checksum_sha256 = ac_checksum
    video.size_bytes = len(content)
    video.mime_type = file.content_type or 'video/mp4'
    video.original_name = file.filename
    video.ac_storage_url = 'N/A'  # TODO? object_url(ac_key)

    # Transition job to running and spawn inference
    job.status = JobStatus.running
    job.started_at = timestamp_now()

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

    await db.flush()

    return {'ok': True}


@router.get('', response_model=JobListResponse)
async def list_jobs(
    status_filter: Optional[str] = Query(None, alias='status'),
    updated_after: Optional[datetime] = Query(None),
    db: DatabaseAccessor = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    with timeit('list_jobs'):
        rows = await db.get_jobs_by_user(user, status_filter, updated_after)
    items = [
        JobSummaryItem(
            id=str(job.id),
            video_id=str(job.video_id),
            status=job.status.value,
            created_at=job.created_at,
            updated_at=job.updated_at,
            original_file_path=video.original_file_path,
            original_checksum_sha256=video.original_checksum_sha256,
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

    return JobDetail(
        id=str(job.id),
        video_id=str(job.video_id),
        status=job.status.value,
        created_at=job.created_at,
        updated_at=job.updated_at,
        tracks=job.tracks,
        original_file_path=video.original_file_path,
        original_checksum_sha256=video.original_checksum_sha256,
        dominant_orientation=job.dominant_orientation,
    )


@router.delete('/{job_id}')
async def delete_job(job_id: str, db: DatabaseAccessor = Depends(get_db), user: User = Depends(authenticate_user)):
    job = await db.get_job_by_id_and_user(job_id, user)
    if job is None:
        raise HTTPException(status_code=404, detail='Not found')

    job.deleted_at = timestamp_now()
    await db.flush()  # Flush to update the job

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

    return {'ok': True}


class JobsCompleteRequest(BaseModel):
    tracks: list[Any]
    dominant_orientation: int
    status: Literal['succeeded', 'failed']


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

    job.tracks = payload.tracks
    job.dominant_orientation = payload.dominant_orientation
    job.status = JobStatus(payload.status)
    job.finished_at = timestamp_now()
    await db.flush()  # Flush to update the job

    return {'ok': True}
