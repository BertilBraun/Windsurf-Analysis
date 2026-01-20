from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.firebase_auth import User, get_current_user
from config import settings
from models import JobPatch, JobStatus, TrackResult, StabilizationTransform
from repos.jobs_repo import JobsRepo
from repos.reports_repo import ReportType, ReportsRepo
from repos.user_jobs_repo import UserJobsRepo
from repos.user_repo import UserRepo
from db.firestore_client import now


router = APIRouter(prefix='/jobs', tags=['jobs'])
jobs_repo = JobsRepo()
user_jobs_repo = UserJobsRepo()
reports_repo = ReportsRepo()
user_repo = UserRepo()


class JobCreateRequest(BaseModel):
    original_checksum_sha256: str = Field(min_length=8)
    original_file_size_bytes: int = Field(ge=0)
    original_file_mime_type: str = Field(min_length=1)


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobUploadCompleteRequest(BaseModel):
    object_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    mime_type: str = 'video/mp4'
    yolo_model: str = Field(min_length=1)


class JobsBulkDeleteRequest(BaseModel):
    job_ids: list[str] = Field(default_factory=list)


class JobSummaryItem(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    original_checksum_sha256: str
    dominant_orientation: int = 0


class JobListResponse(BaseModel):
    jobs: list[JobSummaryItem]


class JobDetail(JobSummaryItem):
    tracks: list[TrackResult]
    stabilization_transforms: list[StabilizationTransform]


class ReportRequest(BaseModel):
    message: str
    type: ReportType


def _require_owned(user: User, job_id: str) -> None:
    association = user_jobs_repo.get_user_job(user.uid, job_id)
    if association is None or association.deleted_at is not None:
        raise HTTPException(status_code=404, detail='Not found')


@router.post('', response_model=JobCreateResponse)
def create_job(payload: JobCreateRequest, user: User = Depends(get_current_user)):
    existing = jobs_repo.get_job_by_checksum(payload.original_checksum_sha256)
    if existing is not None:
        user_jobs_repo.create_user_job(user.uid, existing.job_id)
        jobs_repo.touch_job_accessed(existing.job_id)
        return JobCreateResponse(job_id=existing.job_id, status=existing.status)

    user_record = user_repo.get_user(user.uid)
    if user_record.processed_jobs_count >= user_record.max_jobs:
        raise HTTPException(status_code=403, detail={'code': 'quota_exceeded', 'message': 'Job quota exceeded'})

    job_record = jobs_repo.create_job(
        original_checksum_sha256=payload.original_checksum_sha256,
        original_file_size_bytes=payload.original_file_size_bytes,
        original_file_mime_type=payload.original_file_mime_type,
    )
    user_jobs_repo.create_user_job(user.uid, job_record.job_id)
    return JobCreateResponse(job_id=job_record.job_id, status=job_record.status)


def _gs_uri(bucket: str, object_path: str) -> str:
    clean = object_path.lstrip('/')
    return f'gs://{bucket}/{clean}'


def _trigger_modal_start(job_id: str, *, source_gs_uri: str, upright_gs_uri: str, yolo_model: str) -> None:
    if not settings.modal_shared_secret:
        raise RuntimeError('MODAL_SHARED_SECRET is not configured')
    if not settings.modal_trigger_base_url:
        raise RuntimeError('MODAL_TRIGGER_BASE_URL is not configured')

    import json
    import urllib.request

    url = f'{settings.modal_trigger_base_url}/api/v1/internal/jobs/{job_id}/start'
    body = json.dumps(
        {'source_gs_uri': source_gs_uri, 'upright_gs_uri': upright_gs_uri, 'yolo_model': yolo_model}
    ).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        method='POST',
        headers={
            'Content-Type': 'application/json',
            settings.modal_secret_header: settings.modal_shared_secret,
        },
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        if getattr(res, 'status', 200) >= 400:
            raise RuntimeError(f'Modal trigger failed: HTTP {res.status}')


@router.post('/{job_id}/upload/complete')
def upload_complete(job_id: str, payload: JobUploadCompleteRequest, user: User = Depends(get_current_user)):
    _require_owned(user, job_id)

    job = jobs_repo.get_job(job_id)
    if job.status != JobStatus.uploading:
        # Idempotent behavior: if the job already moved on, don't error.
        return {'ok': True, 'status': job.status.value}

    if not payload.object_path.startswith(f'uploads/{user.uid}/{job_id}'):
        raise HTTPException(status_code=400, detail='Invalid object_path')
    if job.size_bytes != payload.size_bytes:
        raise HTTPException(status_code=400, detail='Size mismatch')
    if job.mime_type != payload.mime_type:
        raise HTTPException(status_code=400, detail='Mime type mismatch')

    bucket = settings.firebase_storage_bucket
    if not bucket:
        raise HTTPException(status_code=500, detail='FIREBASE_STORAGE_BUCKET is not configured')

    source_gs_uri = _gs_uri(bucket, payload.object_path)
    upright_object_path = f'processed/{user.uid}/{job_id}_upright.mp4'
    upright_gs_uri = _gs_uri(bucket, upright_object_path)

    # Persist upload metadata and move to "starting" before triggering Modal.
    jobs_repo.update_job(
        job_id,
        JobPatch(
            size_bytes=payload.size_bytes,
            mime_type=payload.mime_type,
            ac_storage_url=source_gs_uri,
            uploaded_at=now(),
            status=JobStatus.starting,
            started_at=now(),
            error_message=None,
        ),
    )

    try:
        _trigger_modal_start(
            job_id, source_gs_uri=source_gs_uri, upright_gs_uri=upright_gs_uri, yolo_model=payload.yolo_model
        )
    except Exception as e:
        jobs_repo.update_job(job_id, JobPatch(status=JobStatus.failed, error_message=str(e)))
        raise HTTPException(status_code=502, detail=f'Failed to start processing: {e}')

    return {'ok': True, 'status': JobStatus.starting.value}


@router.post('/bulk-delete')
def bulk_delete_jobs(payload: JobsBulkDeleteRequest, user: User = Depends(get_current_user)):
    job_ids = [j for j in payload.job_ids if isinstance(j, str) and j]
    # Safety guard: avoid huge requests.
    if len(job_ids) > 500:
        raise HTTPException(status_code=400, detail='Too many job_ids')

    # Only delete associations the user actually has; ignore unknown ids to keep it idempotent.
    allowed: list[str] = []
    for job_id in job_ids:
        assoc = user_jobs_repo.get_user_job(user.uid, job_id)
        if assoc is None or assoc.deleted_at is not None:
            continue
        allowed.append(job_id)

    deleted = user_jobs_repo.mark_user_jobs_deleted(user.uid, allowed)
    return {'ok': True, 'deleted': deleted}


@router.get('/{job_id}', response_model=JobDetail)
def get_job(job_id: str, user: User = Depends(get_current_user)):
    _require_owned(user, job_id)

    job = jobs_repo.get_job(job_id)
    results = jobs_repo.get_results(job_id)

    if job.status != JobStatus.succeeded or results is None:
        raise HTTPException(status_code=405, detail='Job not completed')

    jobs_repo.touch_job_accessed(job_id)

    return JobDetail(
        id=job_id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        original_checksum_sha256=job.original_checksum_sha256,
        dominant_orientation=job.dominant_orientation,
        tracks=results.tracks,
        stabilization_transforms=results.stabilization_transforms,
    )


@router.delete('/{job_id}')
def delete_job(job_id: str, user: User = Depends(get_current_user)):
    _require_owned(user, job_id)
    user_jobs_repo.mark_user_job_deleted(user.uid, job_id)
    # TODO: if no other user jobs point at this job, delete the job? Recursive delete to also delete the results document
    return {'ok': True}


@router.post('/{job_id}/report')
def report_job(job_id: str, payload: ReportRequest, user: User = Depends(get_current_user)):
    _require_owned(user, job_id)
    reports_repo.add_report(user.uid, job_id, payload.type, payload.message)
    return {'ok': True}
