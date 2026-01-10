from __future__ import annotations

import json
import os
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

_gcs_client: storage.Client | None = None


def get_gcs_client() -> storage.Client:
    """Create (and cache) a GCS client.

    Prefers explicit service account JSON via `GCP_SERVICE_ACCOUNT_JSON` to avoid
    relying on ambient credentials inside Modal containers.
    """

    global _gcs_client
    if _gcs_client is not None:
        return _gcs_client

    raw = os.environ.get('GCP_SERVICE_ACCOUNT_JSON')
    if raw:
        info = json.loads(raw)
        credentials = service_account.Credentials.from_service_account_info(info)
        _gcs_client = storage.Client(project=info.get('project_id'), credentials=credentials)
        return _gcs_client

    _gcs_client = storage.Client()
    return _gcs_client


def parse_gs_uri(gs_uri: str) -> tuple[str, str]:
    if not gs_uri.startswith('gs://'):
        raise ValueError(f'Invalid gs uri: {gs_uri}')
    rest = gs_uri[len('gs://') :]
    bucket, _, key = rest.partition('/')
    if not bucket or not key:
        raise ValueError(f'Invalid gs uri: {gs_uri}')
    return bucket, key


def download_gs_uri(gs_uri: str, *, dest_path: str) -> str:
    bucket_name, key = parse_gs_uri(gs_uri)
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(dest_path)
    return dest_path


def upload_file_to_gs_uri(local_path: str, *, gs_uri: str, content_type: str | None = None) -> None:
    bucket_name, key = parse_gs_uri(gs_uri)
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)
    blob.upload_from_filename(local_path, content_type=content_type)

