from __future__ import annotations

import hashlib
import mimetypes
import uuid
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import authenticate_user, get_db
from .config import settings
from .db import init_db
from .models import Job, JobStatus, Report, ReportType, User, Video
from .s3 import object_url, s3_client
from .schemas import (
    ChecksumPreflightRequest,
    ChecksumPreflightResponse,
    ErrorResponse,
    JobCreateUploadResponse,
    JobListItem,
    JobListResponse,
    ReportRequest,
)


app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(',')],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.on_event('startup')
async def on_startup():
    await init_db()


@app.post('/v1/videos/checksum', response_model=ChecksumPreflightResponse)
async def videos_checksum(
    payload: ChecksumPreflightRequest, db: AsyncSession = Depends(get_db), user: User = Depends(authenticate_user)
):
    result = await db.execute(select(Video).where(Video.original_checksum_sha256 == payload.original_checksum_sha256))
    existing = result.scalars().first()
    if existing:
        return ChecksumPreflightResponse(exists=True, video_id=str(existing.id))
    return ChecksumPreflightResponse(exists=False)


@app.post('/v1/jobs.upload', response_model=JobCreateUploadResponse, responses={409: {'model': ErrorResponse}})
async def jobs_upload(
    request: Request,
    file: UploadFile = File(...),
    original_file_path: str = Form(...),
    original_checksum_sha256: str = Form(...),
    model: str = Form('yolo-8n'),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    # Quota enforcement
    res = await db.execute(
        select(func.count())
        .select_from(Job)
        .where(
            and_(
                Job.user_id == user.id,
                Job.deleted_at.is_(None),
                Job.status.in_([JobStatus.pending, JobStatus.running, JobStatus.succeeded]),
            )
        )
    )
    count = res.scalar_one()
    if count >= settings.MAX_JOBS_PER_USER:
        raise HTTPException(status_code=403, detail={'code': 'quota_exceeded', 'message': 'Job quota exceeded'})

    # Compute AC checksum while streaming to S3
    s3 = s3_client()
    ac_key_prefix = f'{settings.PREFIX_AC_VIDEOS}{original_checksum_sha256}/ac/'

    # Read file content into memory for simplicity (improve with multipart upload for >50MB)
    content = await file.read()
    ac_checksum = hashlib.sha256(content).hexdigest()

    # Check duplicate by original checksum
    exists = await db.execute(select(Video).where(Video.original_checksum_sha256 == original_checksum_sha256))
    existing_video = exists.scalars().first()
    if existing_video:
        raise HTTPException(
            status_code=409,
            detail={
                'error': {
                    'code': 'duplicate_original',
                    'message': 'Video already exists',
                    'video_id': str(existing_video.id),
                }
            },
        )

    ext = mimetypes.guess_extension(file.content_type or 'video/mp4') or '.mp4'
    ac_key = f'{ac_key_prefix}{ac_checksum}{ext}'
    s3.put_object(Bucket=settings.S3_BUCKET, Key=ac_key, Body=content, ContentType=file.content_type or 'video/mp4')

    video = Video(
        original_checksum_sha256=original_checksum_sha256,
        original_file_path=original_file_path,
        ac_checksum_sha256=ac_checksum,
        size_bytes=len(content),
        mime_type=file.content_type or 'video/mp4',
        original_name=file.filename,
        ac_storage_url=object_url(ac_key),
    )
    db.add(video)
    await db.flush()

    job = Job(user_id=user.id, video_id=video.id, model=model, status=JobStatus.pending)
    db.add(job)
    await db.flush()
    await db.commit()

    # Trigger Modal run via HTTP
    # Keep simple: POST to MODAL_INVOKE_URL with payload
    import httpx

    webhook_secret = settings.BACKEND_WEBHOOK_SECRET
    complete_url = f'{settings.BACKEND_PUBLIC_BASE_URL}/v1/jobs/{job.id}/complete?secret={webhook_secret}'
    payload = {
        'job_id': str(job.id),
        'ac_storage_url': video.ac_storage_url,
        'model': model,
        'complete_webhook': complete_url,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        await client.post(settings.MODAL_INVOKE_URL, json=payload)

    return JobCreateUploadResponse(job_id=str(job.id), status=job.status.value)


@app.get('/v1/jobs', response_model=JobListResponse)
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
        JobListItem(
            id=str(r.id),
            video_id=str(r.video_id),
            model=r.model,
            status=r.status.value,
            created_at=r.created_at,
            updated_at=r.updated_at,
            results_json_url=r.results_json_url,
        )
        for r in rows
    ]
    return JobListResponse(jobs=items)


@app.get('/v1/jobs/{job_id}', response_model=JobListItem)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(authenticate_user)):
    row = (
        (await db.execute(select(Job).where(and_(Job.id == uuid.UUID(job_id), Job.user_id == user.id))))
        .scalars()
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail='Not found')
    # Touch last_accessed_at for associated video
    await db.execute(update(Video).where(Video.id == row.video_id).values(last_accessed_at=func.now()))
    await db.commit()
    return JobListItem(
        id=str(row.id),
        video_id=str(row.video_id),
        model=row.model,
        status=row.status.value,
        created_at=row.created_at,
        updated_at=row.updated_at,
        results_json_url=row.results_json_url,
    )


@app.delete('/v1/jobs/{job_id}')
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(authenticate_user)):
    row = (
        (await db.execute(select(Job).where(and_(Job.id == uuid.UUID(job_id), Job.user_id == user.id))))
        .scalars()
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail='Not found')
    row.deleted_at = datetime.utcnow()
    await db.commit()
    return {'ok': True}


@app.post('/v1/jobs/{job_id}/report')
async def report_job(
    job_id: str, payload: ReportRequest, db: AsyncSession = Depends(get_db), user: User = Depends(authenticate_user)
):
    row = (
        (await db.execute(select(Job).where(and_(Job.id == uuid.UUID(job_id), Job.user_id == user.id))))
        .scalars()
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail='Not found')
    rep = Report(job_id=row.id, type=ReportType(payload.type), message=payload.message)
    db.add(rep)
    await db.commit()
    return {'ok': True}


@app.post('/v1/jobs/{job_id}/complete')
async def jobs_complete(job_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    secret = request.query_params.get('secret')
    if not secret or secret != settings.BACKEND_WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail='invalid secret')

    body = await request.json()
    # Expect: { "status": "succeeded|failed", "results_json": { ... } }
    row = (
        await db.execute(select(Job, Video).join(Video, Video.id == Job.video_id).where(Job.id == uuid.UUID(job_id)))
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail='Job not found')
    job_row, video_row = row

    # Upload results JSON to S3
    import json

    s3 = s3_client()
    results_key = f'{settings.PREFIX_RESULTS_JSON}{video_row.original_checksum_sha256}/{job_row.model}/result.json'
    s3.put_object(
        Bucket=settings.S3_BUCKET,
        Key=results_key,
        Body=json.dumps(body.get('results_json', {})),
        ContentType='application/json',
    )

    job_row.results_json_url = object_url(results_key)
    job_row.status = JobStatus(body.get('status', 'succeeded'))
    job_row.finished_at = datetime.utcnow()
    await db.commit()
    return {'ok': True}
