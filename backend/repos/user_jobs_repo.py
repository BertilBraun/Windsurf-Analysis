from __future__ import annotations

from google.cloud import firestore

from db.firestore_client import db, now, user_jobs
from models import UserJobRecord


class UserJobsRepo:
    def _doc(self, user_id: str, job_id: str) -> firestore.DocumentReference:
        return user_jobs.document(f'{user_id}_{job_id}')

    # UserJob
    def create_user_job(self, user_id: str, job_id: str) -> None:
        self._doc(user_id, job_id).set(
            UserJobRecord(user_id=user_id, job_id=job_id, deleted_at=None).model_dump(mode='json')
        )

    def get_user_job(self, user_id: str, job_id: str) -> UserJobRecord | None:
        snap = self._doc(user_id, job_id).get()
        if not snap.exists:
            return None
        return UserJobRecord.model_validate(snap.to_dict() or {})

    def mark_user_job_deleted(self, user_id: str, job_id: str) -> None:
        self._doc(user_id, job_id).set(
            UserJobRecord(user_id=user_id, job_id=job_id, deleted_at=now()).model_dump(mode='json')
        )

    def delete_all_for_user(self, user_id: str) -> int:
        """Hard-delete all user_jobs docs for a user. Returns count deleted."""
        deleted = 0
        batch = db.batch()
        for snap in user_jobs.where('user_id', '==', user_id).stream():
            batch.delete(snap.reference)
            deleted += 1
            # Firestore batch limit is 500 operations; keep margin.
            if deleted % 450 == 0:
                batch.commit()
                batch = db.batch()
        if deleted % 450 != 0:
            batch.commit()
        # TODO: if no other user jobs point at this job, delete the job? Recursive delete to also delete the results document
        return deleted

    def get_user_ids_for_job(self, job_id: str) -> list[str]:
        """Returns user_ids associated with a job (excluding soft-deleted associations)."""
        user_ids: list[str] = []
        for snap in user_jobs.where('job_id', '==', job_id).stream():
            record = UserJobRecord.model_validate(snap.to_dict() or {})
            if record.deleted_at is None:
                user_ids.append(record.user_id)
        return user_ids
