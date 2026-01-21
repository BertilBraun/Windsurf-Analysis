from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

from google.cloud import storage
from google.oauth2 import service_account


@lru_cache(maxsize=1)
def get_gcs_client() -> storage.Client:
    raw = os.environ.get('GCP_SERVICE_ACCOUNT_JSON')
    if raw:
        info = json.loads(raw)
        credentials = service_account.Credentials.from_service_account_info(info)
        return storage.Client(project=info.get('project_id'), credentials=credentials)
    return storage.Client()


def _require_safe_object_name(name: str) -> str:
    clean = name.lstrip('/')
    if not clean or clean.startswith('/') or '..' in clean or '\\' in clean:
        raise ValueError(f'Invalid object name: {name!r}')
    return clean


def upload_json(*, bucket: str, object_name: str, payload: dict[str, Any]) -> None:
    client = get_gcs_client()
    obj = _require_safe_object_name(object_name)
    blob = client.bucket(bucket).blob(obj)
    blob.cache_control = 'no-store'
    blob.upload_from_string(
        json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
        content_type='application/json; charset=utf-8',
    )


def download_json(*, bucket: str, object_name: str) -> dict[str, Any] | None:
    client = get_gcs_client()
    obj = _require_safe_object_name(object_name)
    blob = client.bucket(bucket).blob(obj)
    if not blob.exists():
        return None
    data = blob.download_as_bytes()
    return json.loads(data.decode('utf-8'))

