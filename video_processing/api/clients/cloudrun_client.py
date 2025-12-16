from __future__ import annotations

from dataclasses import dataclass

import httpx

from config import settings


class CloudRunError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f'Cloud Run error {status_code}: {body}')
        self.status_code = status_code
        self.body = body


@dataclass
class CloudRunClient:
    timeout_seconds: float = 10.0

    def _base_headers(self, authorization: str | None) -> dict[str, str]:
        if not settings.modal_shared_secret:
            raise RuntimeError('MODAL_SHARED_SECRET is not configured in Modal environment')
        if not settings.cloud_run_base_url:
            raise RuntimeError('CLOUD_RUN_BASE_URL is not configured in Modal environment')
        headers: dict[str, str] = {
            'X-Modal-Secret': settings.modal_shared_secret,
            'Content-Type': 'application/json',
        }
        if authorization:
            headers['Authorization'] = authorization
        return headers

    def _post(self, path: str, json: dict | None = None, authorization: str | None = None) -> dict:
        url = f'{settings.cloud_run_base_url}{path}'
        with httpx.Client(timeout=self.timeout_seconds) as client:
            res = client.post(url, headers=self._base_headers(authorization), json=json)
        if res.status_code >= 400:
            raise CloudRunError(res.status_code, res.text)
        return res.json() if res.text else {}

    def verify_job(self, job_id: str, authorization: str, required_statuses: list[str] | None = None) -> None:
        self._post(
            f'/internal/jobs/{job_id}/verify',
            json={'required_statuses': required_statuses} if required_statuses is not None else {},
            authorization=authorization,
        )

    def mark_uploaded(
        self,
        job_id: str,
        authorization: str,
        *,
        ac_checksum_sha256: str,
        size_bytes: int,
        mime_type: str,
        ac_storage_url: str,
    ) -> None:
        self._post(
            f'/internal/jobs/{job_id}/uploaded',
            json={
                'ac_checksum_sha256': ac_checksum_sha256,
                'size_bytes': size_bytes,
                'mime_type': mime_type,
                'ac_storage_url': ac_storage_url,
            },
            authorization=authorization,
        )

    def update_status(
        self, job_id: str, *, status: str, error_message: str | None = None, authorization: str | None = None
    ) -> None:
        self._post(
            f'/internal/jobs/{job_id}/status',
            json={'status': status, 'error_message': error_message},
            authorization=authorization,
        )

    def set_results(
        self,
        job_id: str,
        *,
        status: str,
        results: dict | None,
        error_message: str | None = None,
    ) -> None:
        self._post(
            f'/internal/jobs/{job_id}/results',
            json={'status': status, 'results': results, 'error_message': error_message},
        )
