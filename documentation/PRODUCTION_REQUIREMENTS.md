# Production Requirements — GPU Inference Service with React Frontend (v1.0)

Status: **historical design spec**.

This document describes an earlier production plan that assumed:

- HTTP Basic Auth
- Neon Postgres
- S3-compatible object storage (e.g., Cloudflare R2)

The currently implemented MVP in this repo uses **Firebase Auth + Firestore + Firebase Storage** instead. See:

- `documentation/README.md` (what’s implemented)
- `documentation/DEPLOYMENT.md` (current deployment)

---

## 1) Purpose & Goals

Turn the existing CLI/Python-visualization prototype into a **production-ready web application** with:

* **GPU-backed inference service** on **Modal** (serverless, scale-to-zero).
* **React frontend** for upload, tracking, playback, per-user deletion, and fault reporting.
* **Basic authentication** (username/password) for a small, controlled user base.
* **Relational database** (Neon PostgreSQL) holding **users**, a unified **jobs** table (per-user access control), and a dedupe **videos** table.
* **Object storage** (S3-compatible, e.g., Cloudflare R2) for the **analysis copy (AC)** videos, results JSON.
* **Per‑user job quotas** (soft limit of **5 jobs**), with an upsell message.
* **Checksum deduplication** to avoid reprocessing identical videos.

**Media defaults (configurable):** AC target **1920×1080**, **25 fps**, **audio stripped**, H.264.

---

## 2) System Overview

### Components

1. **Frontend (React)** — Upload UI, job feed with 1s polling, video player, “Delete for me”, and “Report mistake”.
2. **Backend API (FastAPI)** — Auth, quota enforcement, **single-call job creation with upload**, dedupe checks, triggers Modal runs, receives webhooks.
3. **Modal GPU Functions** — On-demand GPU inference; **no DB polling**; call backend webhook on completion.
4. **PostgreSQL (Neon)** — Users, **unified jobs**, videos.
5. **Object Storage (R2/S3)** — AC videos and result artifacts (JSON).

### Event-Driven Flow (scale-to-zero)

```
User → React → Backend API → (Neon + Object Storage)
                             ↓           ↑
                         Modal GPU run ──┘ (backend triggers; Modal updates job status on completion)
```

**Reverse proxy:** Not required when deploying on Vercel/Render/Modal; those platforms provide managed HTTPS and routing.

---

## 2A) Bandwidth & Media Strategy (Downscale-on-Upload + Local-Original Playback)

* Clients create an **analysis copy (AC)** before upload:

  * Default **1080p @ 25 fps**, H.264, **no audio** (configurable `AC_MAX_RES`, `AC_TARGET_FPS`, `AC_STRIP_AUDIO`).
  * Compute **SHA‑256 of the original** (dedupe key) and **SHA‑256 of AC** (integrity check).
  * Upload **AC** only; originals remain local on the user’s device.
* Inference runs on the AC; results reference the AC timeline.
* **Playback:** Prefer local original. Player verifies checksum, then overlays detections at native resolution/FPS. Fallback is streaming AC + overlays.

---

## 3) Authentication & Authorization (Basic)

* HTTP Basic over HTTPS on **every request**.
* Passwords: bcrypt/Argon2 hashes in Neon.
* Rate-limit login attempts; lockouts/logging for abuse.
* Access control: all job reads are restricted to `jobs.user_id`.

---

## 4) Quotas & Abuse Prevention

* `MAX_JOBS_PER_USER = 5` (env-configurable).
* Enforce on job creation; return 403 `quota_exceeded` with contact info.
* Upload/file size limits (e.g., 2 GiB AC maximum; configurable).

---

## 5) Data Model (PostgreSQL)

**Design:** Merge the previous `tasks`, `task_artifacts`, and per-user `jobs` into a **single `jobs` table** that holds execution state and artifact pointers. Keep a separate `videos` table keyed by the **original checksum** to deduplicate inputs and store AC metadata.

### Entities & Relationships

```
users 1—* jobs *—1 videos
reports — per job
```

### Tables (DDL Sketch)

```sql
CREATE TYPE job_status AS ENUM ('pending','running','succeeded','failed','canceled');
CREATE TYPE report_type AS ENUM ('missed_detection', 'false_association', 'other');

CREATE TABLE users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  username citext UNIQUE NOT NULL,
  password_hash text NOT NULL,
  last_active_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- One row per distinct original video; stores the Analysis Copy (AC)
CREATE TABLE videos (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  original_checksum_sha256 char(64) UNIQUE NOT NULL,
  original_file_path text NOT NULL,
  size_bytes bigint NOT NULL,                 -- size of AC
  mime_type text NOT NULL,                    -- AC mime (e.g., video/mp4)
  original_name text,                         -- client-provided
  ac_storage_url text NOT NULL,               -- s3://... (AC location)
  uploaded_at timestamptz NOT NULL DEFAULT now(),
  last_accessed_at timestamptz NOT NULL DEFAULT now()
);

-- Unified per-user job + execution state + artifact pointers
CREATE TABLE jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  video_id uuid NOT NULL REFERENCES videos(id) ON DELETE RESTRICT,
  model text NOT NULL,
  status job_status NOT NULL DEFAULT 'pending',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  finished_at timestamptz,
  error_message text,
  -- Artifact pointers (store in object storage, not in DB)
  results_json_url text,                      -- e.g., s3://.../result.json
  deleted_at timestamptz,
  UNIQUE (user_id, video_id, model)
);

-- Feedback
CREATE TABLE reports (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id uuid NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  type report_type NOT NULL,
  message text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Helpful indexes
CREATE INDEX idx_jobs_user_created ON jobs(user_id, created_at DESC);
CREATE INDEX idx_videos_original ON videos(original_checksum_sha256);
CREATE INDEX idx_videos_last_accessed ON videos(last_accessed_at);
```

**Notes**

* **Results JSON is not stored in Postgres**; only URLs are stored in `jobs.*_url`. JSONs are typically hundreds of KB.
* `videos.original_checksum_sha256` is the dedupe key across the system.
* `last_accessed_at` updates when jobs/artifacts are viewed (debounced to avoid write storms).

---

## 6) Storage Layout

**Buckets / prefixes (configurable):**

* `ac-videos/` — `original-checksum/ac/<ac-checksum>.mp4` (Analysis Copy)
* `results-json/` — `original-checksum/model/result.json`
* `logs/` — per-job logs if needed

**Lifecycle & Cost Controls**

* Track **`videos.last_accessed_at`** and update on access.
* Retention sweeper removes AC and/or artifacts if not accessed for **N days** (default 10). Configurable grace periods for recently active users.

---

## 7) API Design (REST over HTTPS)

### Conventions

* **Auth:** HTTP Basic on every request.
* **Uploads:** `multipart/form-data` for AC uploads.
* **Errors:** `{ "error": { "code": "string", "message": "human readable" } }`

### Endpoints

#### (Optional) Preflight — POST /v1/videos/checksum

Check if an **original checksum** already exists so the client can **avoid uploading**.
Input:

```json
{ "original_checksum_sha256": "<64-hex>" }
```

Response:

```json
{ "exists": true, "video_id": "uuid" }
```

#### Create Job + Upload (single call) — POST /v1/jobs.upload

Creates a job **and** uploads the AC in one request. The server **asserts** that the original checksum does **not** already exist to prevent duplicate uploads.

Request (multipart/form-data):

* `file`: AC video (mp4)
* `original_file_path`: path to the original file
* `original_checksum_sha256`: 64-hex of the original file

Server behavior:

1. **Enforce quota** for current user.
4. Persist AC to object storage; create `videos` row.
5. Create **jobs** row for the user (status `uploading`).
6. Trigger **Modal** run (see §8) and return `201 { job_id, status }`.

Response 201:

```json
{ "job_id": "uuid", "status": "pending" }
```

Response:

```json
{ "job_id": "uuid", "status": "pending" }
```

#### List Jobs — GET /v1/jobs?status={status}&updated_after={updated_after}

Returns non-deleted jobs for the authenticated user.

#### Job Detail — GET /v1/jobs/{job\_id}

Returns job state and artifact pointers; touches `videos.last_accessed_at`.

#### Delete (soft) — DELETE /v1/jobs/{job\_id}

Soft-delete for that user only.

#### Report Mistake — POST /v1/jobs/{job\_id}/report

`{ "message": "Object missed at 00:12:31", "type": "missed_detection" }`

#### Job Complete (Modal → Backend) — POST /v1/jobs/{job\_id}/complete?secret={secret}

* Signed with `secret=BACKEND_SECRET`.
* Body includes result metadata + storage URLs.
* Backend updates `jobs.status`, `results_json_url` timestamps.

---

## 8) Inference Execution (Modal)

* **Trigger:** Backend calls the Modal function with `{ job_id, ac_storage_url, model, secret }` and a one-time signed webhook secret.
* **Run:** Modal pulls the AC, runs GPU inference.
* **Complete:** Modal calls /v1/jobs/{job\_id}/complete?secret={secret}. Backend verifies signature, updates the `jobs` row, and sets `status` accordingly and writes the `result.json` to object storage.
* **Retries:** Configure Modal retries with idempotent webhook processing (use `job_id` as idempotency key).
* **Concurrency:** Controlled at Modal; scale to zero when idle.

---

## 9) Frontend Requirements

* **Login**: Basic Auth; store credentials in memory.
* **Upload flow**:

  1. Call `/v1/videos/checksum` to avoid uploading duplicates.
  2. If not exists, **transcode AC** (1080p/25fps/no-audio), compute checksums.
  3. `POST /v1/jobs.upload` (single call) to upload AC and create the job.
* **Job feed**: Poll `/v1/jobs` every 1s for pending/running; show play icon when `succeeded`. If no jobs are pending/running, slow down polling to 10s.
* **Player**: Prefer **local-original overlay** (verify checksum). Fallback to download AC + overlays
* **Controls**: Delete (soft only remove from feed), Report mistake (prominent over delete - for us to detect issues and fix them).

---

## 10) Data Lifecycle - Not necessary for MVP

* **Backups:** Neon scheduled backups; enable bucket versioning if desired.
* **Retention:**

  * `VIDEO_DELETE_IF_NOT_ACCESSED_DAYS` (default 10): remove AC and results if beyond threshold.
  * `USER_INACTIVE_GRACE_DAYS` (default 30): optional grace if user active recently.
  * `MIN_ARTIFACT_RETENTION_DAYS` (default 7) safety floor.
* **Sweeper:** Daily scheduled task (serverless cron) to evaluate candidates, delete objects, and log actions.

---

## 11) Security

* **HTTPS everywhere** (Vercel/Render/Modal manage TLS).
* **Basic Auth**; strong bcrypt/Argon2 params.
* **Rate limits** on login.
* **Signed URLs** for object storage access; never expose bucket creds to the browser.
* **CORS** locked to frontend origin.
