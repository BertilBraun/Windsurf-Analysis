from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class JobStatus(str, Enum):
    pending = 'pending'
    starting = 'starting'
    orientation = 'orientation'
    stabilization = 'stabilization'
    detection = 'detection'
    appearance = 'appearance'
    tracking = 'tracking'
    succeeded = 'succeeded'
    failed = 'failed'
    canceled = 'canceled'


class _FirestoreModel(BaseModel):
    """Pydantic models used as the strict-ish boundary after Firestore reads.

    - Extra fields are ignored (we don't care).
    - Missing required fields will raise a ValidationError.
    """

    model_config = ConfigDict(extra='ignore', frozen=True)


class JobResults(_FirestoreModel):
    tracks: list[Any]
    dominant_orientation: int
    stabilization_transforms: list[Any]


class JobRecord(_FirestoreModel):
    job_id: str
    original_checksum_sha256: str
    ac_checksum_sha256: str
    size_bytes: int
    mime_type: str
    ac_storage_url: str
    uploaded_at: datetime | None  # required key, value may be None
    last_accessed_at: datetime | None  # required key, value may be None
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None  # required key, value may be None
    finished_at: datetime | None  # required key, value may be None
    error_message: str | None  # required key, value may be None
    deleted_at: datetime | None  # required key, value may be None
    dominant_orientation: int


class UserRecord(_FirestoreModel):
    user_id: str
    email: str
    processed_jobs_count: int
    max_jobs: int
    created_at: datetime | None  # required key, value may be None
    last_active_at: datetime | None  # required key, value may be None
    terms_accepted_at: datetime | None = None
    privacy_accepted_at: datetime | None = None
    marketing_consent: bool | None = None
    marketing_consent_at: datetime | None = None


class UserJobRecord(_FirestoreModel):
    user_id: str
    job_id: str
    deleted_at: datetime | None  # required key, value may be None


class JobPatch(_FirestoreModel):
    """Partial update payload for Firestore writes.

    Only fields explicitly set on the model are included in `to_firestore_fields()`.
    This allows "clear the field" via `error_message=None` without requiring a custom UNSET sentinel.
    """

    status: JobStatus | None = None
    error_message: str | None = None
    ac_checksum_sha256: str | None = None
    size_bytes: int | None = None
    mime_type: str | None = None
    ac_storage_url: str | None = None

    # Firestore patches may use firestore.SERVER_TIMESTAMP (not a datetime),
    # so these must accept Any.
    uploaded_at: Any | None = None
    started_at: Any | None = None
    finished_at: Any | None = None
    last_accessed_at: Any | None = None

    dominant_orientation: int | None = None

    def to_firestore_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for name in self.model_fields_set:
            value = getattr(self, name)
            if name == 'status' and isinstance(value, JobStatus):
                fields['status'] = value.value
            else:
                fields[name] = value
        return fields
