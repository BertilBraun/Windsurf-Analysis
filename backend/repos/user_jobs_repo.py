from __future__ import annotations

from google.cloud import firestore

from db.firestore_client import now, user_jobs
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

    def list_job_ids_for_user(self, user_id: str) -> list[str]:
        # Avoid composite-index requirements by filtering deleted_at client-side.
        snaps = user_jobs.where('user_id', '==', user_id).where('deleted_at', '==', None).stream()
        return [UserJobRecord.model_validate(snap.to_dict() or {}).job_id for snap in snaps]
