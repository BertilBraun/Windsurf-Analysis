from __future__ import annotations

from google.cloud import firestore
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Literal

from auth.firebase_auth import User, get_current_user
from auth.internal_auth import require_modal_secret
from models import JobPatch, JobResults, JobStatus
from repos.jobs_repo import JobsRepo
from repos.user_jobs_repo import UserJobsRepo


router = APIRouter(prefix='/internal/jobs', tags=['internal-jobs'], dependencies=[Depends(require_modal_secret)])
jobs_repo = JobsRepo()
user_jobs_repo = UserJobsRepo()


class InternalVerifyRequest(BaseModel):
    required_statuses: list[JobStatus] | None = None


class InternalUploadedRequest(BaseModel):
    ac_checksum_sha256: str
    size_bytes: int = Field(ge=0)
    mime_type: str = 'video/mp4'
    ac_storage_url: str = 'N/A'


class InternalStatusRequest(BaseModel):
    status: JobStatus
    error_message: str | None = None


class JobsCompleteResults(BaseModel):
    tracks: list[Any]
    dominant_orientation: int
    stabilization_transforms: list[Any]


class InternalResultsRequest(BaseModel):
    status: Literal['succeeded', 'failed']
    results: JobsCompleteResults | None = None
    error_message: str | None = None


def _require_owned(user: User, job_id: str) -> None:
    assoc = user_jobs_repo.get_user_job(user.uid, job_id)
    if assoc is None or assoc.deleted_at is not None:
        raise HTTPException(status_code=404, detail='Not found')


@router.post('/{job_id}/verify')
def verify_job(job_id: str, payload: InternalVerifyRequest, user: User = Depends(get_current_user)):
    _require_owned(user, job_id)
    required = payload.required_statuses
    if required:
        job = jobs_repo.get_job(job_id)
        if job.status not in required:
            raise HTTPException(status_code=409, detail='Job not in allowed state')
    return {'ok': True}


@router.post('/{job_id}/uploaded')
def mark_uploaded(job_id: str, payload: InternalUploadedRequest, user: User = Depends(get_current_user)):
    _require_owned(user, job_id)
    job = jobs_repo.get_job(job_id)
    if job.status != JobStatus.pending:
        raise HTTPException(status_code=409, detail='Job not in a state that accepts uploads')
    jobs_repo.update_job(
        job_id,
        JobPatch(
            ac_checksum_sha256=payload.ac_checksum_sha256,
            size_bytes=payload.size_bytes,
            mime_type=payload.mime_type,
            ac_storage_url=payload.ac_storage_url,
            uploaded_at=firestore.SERVER_TIMESTAMP,
            status=JobStatus.starting,
            started_at=firestore.SERVER_TIMESTAMP,
        ),
    )
    return {'ok': True}


@router.post('/{job_id}/status')
def update_status(job_id: str, payload: InternalStatusRequest):
    # Modal workers may not have the user's Firebase token; secret-only is sufficient here.
    patch = (
        JobPatch(status=payload.status)
        if 'error_message' not in payload.model_fields_set
        else JobPatch(status=payload.status, error_message=payload.error_message)
    )
    jobs_repo.update_job(job_id, patch)
    return {'ok': True}


@router.post('/{job_id}/results')
def set_results(job_id: str, payload: InternalResultsRequest):
    # Modal workers may not have the user's Firebase token; secret-only is sufficient here.
    if payload.status != 'succeeded' or payload.results is None:
        jobs_repo.update_job(
            job_id,
            JobPatch(
                status=JobStatus.failed,
                error_message=payload.error_message or 'Failed to complete job',
                finished_at=firestore.SERVER_TIMESTAMP,
            ),
        )
        return {'ok': True}

    results = JobResults(
        tracks=payload.results.tracks,
        dominant_orientation=payload.results.dominant_orientation,
        stabilization_transforms=payload.results.stabilization_transforms,
    )
    jobs_repo.set_results(job_id, results)
    jobs_repo.update_job(
        job_id,
        JobPatch(
            status=JobStatus.succeeded,
            error_message=None,
            finished_at=firestore.SERVER_TIMESTAMP,
            dominant_orientation=results.dominant_orientation,
        ),
    )
    return {'ok': True}
