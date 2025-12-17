from __future__ import annotations

from google.cloud import firestore

from db.firestore_client import now, users
from models import UserRecord
from config import settings
from fastapi import HTTPException


class UserRepo:
    def create_user(self, user_id: str, email: str) -> None:
        users.document(user_id).set(
            UserRecord(
                user_id=user_id,
                email=email,
                processed_jobs_count=0,
                max_jobs=settings.max_jobs_per_user_default,
                last_active_at=now(),
                created_at=now(),
            ).model_dump(mode='json')
        )

    def does_user_exist(self, user_id: str) -> bool:
        return users.document(user_id).get().exists

    def update_last_active_at(self, user_id: str) -> None:
        users.document(user_id).set({'last_active_at': firestore.SERVER_TIMESTAMP}, merge=True)

    def get_user(self, user_id: str) -> UserRecord:
        snap = users.document(user_id).get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail='User not found')
        return UserRecord.model_validate(snap.to_dict() or {})

    def delete_user(self, user_id: str) -> None:
        users.document(user_id).delete()

    def increment_processed_jobs_count(self, user_id: str, delta: int = 1) -> None:
        users.document(user_id).set({'processed_jobs_count': firestore.Increment(delta)}, merge=True)
