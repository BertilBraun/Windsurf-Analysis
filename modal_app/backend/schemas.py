from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChecksumPreflightRequest(BaseModel):
    original_checksum_sha256: str = Field(min_length=64, max_length=64)


class ChecksumPreflightResponse(BaseModel):
    exists: bool
    video_id: Optional[str] = None


class JobCreateUploadResponse(BaseModel):
    job_id: str
    status: Literal['pending', 'running', 'succeeded', 'failed', 'canceled']


class ErrorResponse(BaseModel):
    error: dict


class JobListItem(BaseModel):
    id: str
    video_id: str
    model: str
    status: str
    created_at: datetime
    updated_at: datetime
    results_json_url: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobListItem]


class ReportRequest(BaseModel):
    message: str
    type: Literal['missed_detection', 'false_association', 'other']


class CreateUserRequest(BaseModel):
    secret: str
    email: str
    password: str


class CreateUserResponse(BaseModel):
    id: str
    email: str
