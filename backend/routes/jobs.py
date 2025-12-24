from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.firebase_auth import User, get_current_user
from models import JobStatus
from repos.jobs_repo import JobsRepo
from repos.reports_repo import ReportType, ReportsRepo
from repos.user_jobs_repo import UserJobsRepo
from repos.user_repo import UserRepo


router = APIRouter(prefix='/jobs', tags=['jobs'])
jobs_repo = JobsRepo()
user_jobs_repo = UserJobsRepo()
reports_repo = ReportsRepo()
user_repo = UserRepo()


class JobCreateRequest(BaseModel):
    original_checksum_sha256: str = Field(min_length=8)


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


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
    tracks: list[Any]
    stabilization_transforms: list[Any]


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

    job_record = jobs_repo.create_job(payload.original_checksum_sha256)
    user_jobs_repo.create_user_job(user.uid, job_record.job_id)
    return JobCreateResponse(job_id=job_record.job_id, status=job_record.status)


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
