Modal upload-only service.

Env vars expected in Modal secret/env:
- CLOUD_RUN_BASE_URL: e.g. https://<cloudrun-host>
- MODAL_SHARED_SECRET: same value configured on Cloud Run for X-Modal-Secret

This service exposes:
- POST /jobs/{job_id}/upload/init
- POST /jobs/{job_id}/upload/part
- POST /jobs/{job_id}/upload/complete
- POST /jobs/{job_id}/complete

All endpoints require Authorization: Bearer <Firebase ID token>.
