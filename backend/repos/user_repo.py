"""
Repository module for managing user records in Firestore.
"""

from __future__ import annotations

from datetime import datetime

from google.cloud import firestore

from db.firestore_client import now, users
from models import UserRecord
from config import settings
from fastapi import HTTPException


class UserRepo:
    """
    Handles CRUD operations and state updates for UserRecord documents in Firestore.
    """

    def create_user(
        self,
        user_id: str,
        email: str,
        terms_accepted_at: datetime | None = None,
        privacy_accepted_at: datetime | None = None,
        marketing_consent: bool | None = None,
        marketing_consent_at: datetime | None = None,
    ) -> None:
        """
        Creates a new user record with default settings and provided consent information.
        """
        users.document(user_id).set(
            UserRecord(
                user_id=user_id,
                email=email,
                processed_jobs_count=0,
                max_jobs=settings.max_jobs_per_user_default,
                last_active_at=now(),
                created_at=now(),
                terms_accepted_at=terms_accepted_at,
                privacy_accepted_at=privacy_accepted_at,
                marketing_consent=marketing_consent,
                marketing_consent_at=marketing_consent_at,
            ).model_dump(mode='json')
        )

    def does_user_exist(self, user_id: str) -> bool:
        """
        Checks if a user document exists for the given user ID.
        """
        return users.document(user_id).get().exists

    def update_last_active_at(self, user_id: str) -> None:
        """
        Updates the user's last_active_at field using the Firestore server timestamp.
        """
        users.document(user_id).set({'last_active_at': firestore.SERVER_TIMESTAMP}, merge=True)

    def get_user(self, user_id: str) -> UserRecord:
        """
        Retrieves a UserRecord by ID. Raises HTTPException 404 if the user does not exist.
        """
        snap = users.document(user_id).get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail='User not found')
        return UserRecord.model_validate(snap.to_dict() or {})

    def delete_user(self, user_id: str) -> None:
        """
        Deletes a user record from Firestore.
        """
        users.document(user_id).delete()

    def increment_processed_jobs_count(self, user_id: str, delta: int = 1) -> None:
        """
        Atomically increments the processed_jobs_count for the specified user.
        """
        users.document(user_id).set({'processed_jobs_count': firestore.Increment(delta)}, merge=True)

    def update_user_fields(self, user_id: str, fields: dict) -> None:
        """
        Updates specific fields in a user document using a merge operation.
        """
        if not fields:
            return
        users.document(user_id).set(fields, merge=True)
