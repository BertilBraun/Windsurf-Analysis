from __future__ import annotations

from config import settings
from db.firestore_client import jobs, now
from models import JobPatch, JobRecord, JobResults, JobStatus
from google.cloud import firestore
from fastapi import HTTPException
from storage.gcs_json import download_json, upload_json


class JobsRepo:
    def _results_object_name(self, job_id: str) -> str:
        # Keep deterministic and non-user-specific because jobs are keyed by content checksum.
        if '/' in job_id or '\\' in job_id:
            raise ValueError(f'Invalid job id: {job_id!r}')
        return f'results/{job_id}.json'

    def get_job(self, job_id: str) -> JobRecord:
        snap = jobs.document(job_id).get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail='Job not found')
        return JobRecord.model_validate(snap.to_dict() or {})

    def get_job_by_checksum(self, checksum: str) -> JobRecord | None:
        snap = jobs.document(checksum).get()
        if not snap.exists:
            return None
        return JobRecord.model_validate(snap.to_dict() or {})

    def create_job(
        self,
        original_checksum_sha256: str,
        original_file_size_bytes: int,
        original_file_mime_type: str,
    ) -> JobRecord:
        job_id = original_checksum_sha256
        job_record = JobRecord(
            job_id=job_id,
            original_checksum_sha256=original_checksum_sha256,
            size_bytes=original_file_size_bytes,
            mime_type=original_file_mime_type,
            ac_storage_url='N/A',
            uploaded_at=None,
            last_accessed_at=now(),
            status=JobStatus.uploading,
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
        bucket = settings.firebase_storage_bucket
        if not bucket:
            raise HTTPException(status_code=500, detail='FIREBASE_STORAGE_BUCKET is not configured')

        upload_json(
            bucket=bucket,
            object_name=self._results_object_name(job_id),
            payload=job_results.model_dump(mode='json'),
        )

    def get_results(self, job_id: str) -> JobResults | None:
        bucket = settings.firebase_storage_bucket
        if not bucket:
            raise HTTPException(status_code=500, detail='FIREBASE_STORAGE_BUCKET is not configured')

        payload = download_json(bucket=bucket, object_name=self._results_object_name(job_id))
        if payload is None:
            return None
        return JobResults.model_validate(payload)
