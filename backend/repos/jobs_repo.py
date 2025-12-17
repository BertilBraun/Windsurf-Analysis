from __future__ import annotations

import uuid

from db.firestore_client import jobs, now, results
from models import JobPatch, JobRecord, JobResults, JobStatus
from google.cloud import firestore
from fastapi import HTTPException


class JobsRepo:
    # Jobs
    def get_job(self, job_id: str) -> JobRecord:
        snap = jobs.document(job_id).get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail='Job not found')
        return JobRecord.model_validate(snap.to_dict() or {})

    def get_job_by_checksum(self, checksum: str) -> JobRecord | None:
        q = jobs.where('original_checksum_sha256', '==', checksum).limit(1).stream()
        for snap in q:
            return JobRecord.model_validate(snap.to_dict() or {})
        return None

    def create_job(self, original_checksum_sha256: str) -> JobRecord:
        job_id = str(uuid.uuid4())
        job_record = JobRecord(
            job_id=job_id,
            original_checksum_sha256=original_checksum_sha256,
            ac_checksum_sha256='PENDING',
            size_bytes=-1,
            mime_type='video/mp4',
            ac_storage_url='N/A',
            uploaded_at=None,
            last_accessed_at=now(),
            status=JobStatus.pending,
            created_at=now(),
            updated_at=now(),
            started_at=None,
            finished_at=None,
            error_message=None,
            deleted_at=None,
            dominant_orientation=0,
        )
        jobs.document(job_id).set(job_record.model_dump(mode='json'))
        return job_record

    def touch_job_accessed(self, job_id: str) -> None:
        self.update_job(job_id, JobPatch(last_accessed_at=firestore.SERVER_TIMESTAMP))

    def update_job(self, job_id: str, patch: JobPatch) -> None:
        fields = patch.to_firestore_fields()
        fields['updated_at'] = now()
        jobs.document(job_id).set(fields, merge=True)

    # Results
    def set_results(self, job_id: str, job_results: JobResults) -> None:
        results(job_id).set(job_results.model_dump(mode='json'))

    def get_results(self, job_id: str) -> JobResults | None:
        snap = results(job_id).get()
        if not snap.exists:
            return None
        return JobResults.model_validate(snap.to_dict() or {})
