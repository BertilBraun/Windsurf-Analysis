from __future__ import annotations

from google.cloud import firestore
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Literal

from auth.internal_auth import require_modal_secret
from models import JobPatch, JobResults, JobStatus, TrackResult, StabilizationTransform
from repos.jobs_repo import JobsRepo
from repos.user_repo import UserRepo
from repos.user_jobs_repo import UserJobsRepo


router = APIRouter(prefix='/internal/jobs', tags=['internal-jobs'], dependencies=[Depends(require_modal_secret)])
jobs_repo = JobsRepo()
user_jobs_repo = UserJobsRepo()
user_repo = UserRepo()


class InternalStatusRequest(BaseModel):
    status: JobStatus
    error_message: str | None = None


class JobsCompleteResults(BaseModel):
    tracks: list[TrackResult]
    dominant_orientation: int
    stabilization_transforms: list[StabilizationTransform]


class InternalResultsRequest(BaseModel):
    status: Literal['succeeded', 'failed']
    results: JobsCompleteResults | None = None
    error_message: str | None = None


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
    for user_id in user_jobs_repo.get_user_ids_for_job(job_id):
        user_repo.increment_processed_jobs_count(user_id, 1)
    return {'ok': True}
