# Deployment (Firebase Hosting + Cloud Run + Modal)

This document describes the **current** production MVP deployment for GybeLock / Windsurf Analysis:

- **Frontend**: Firebase Hosting (`frontend/`)
- **Backend API**: FastAPI on Google Cloud Run (`backend/`)
- **Data**: Firebase Auth + Firestore + Firebase Storage (GCS-backed)
- **Compute**: Modal (GPU/CPU pipeline in `video_processing/`)

If you’re setting this up from scratch, also read `documentation/FIREBASE_SETUP.md` (it is the “click-by-click” Firebase guide).

---

## Prerequisites

- Python 3.10+ (Modal images use 3.10; local can be newer)
- Node 18+
- Accounts:
  - Firebase + Google Cloud (same project)
  - Modal

---

## Required configuration (backend)

### Backend env vars (`backend/.env` or Cloud Run env)

- `FIREBASE_STORAGE_BUCKET` — e.g. `gybelock-00.appspot.com`
- `MODAL_SHARED_SECRET` — shared secret used for Modal → Cloud Run internal calls (header `X-Modal-Secret`)
- `MODAL_TRIGGER_BASE_URL` — Modal trigger ASGI app base URL (see “Deploy Modal” below)

Optional (recommended):

- `MAX_JOBS_PER_USER` — soft quota (default: `5`)
- `FIRESTORE_DATABASE` — Firestore database id (default: `(default)`)

### CORS origins

Allowed origins are currently hard-coded in `backend/config.py`. Make sure it includes:

- your Firebase Hosting domain (`https://<project>.web.app`)
- your custom domain (if any)
- local dev origins (`http://localhost:5173`, etc.)

---

## Deploy Modal

Modal apps live under `video_processing/` and are deployed via `video_processing/deploy.py`.

1. Install the Modal CLI and authenticate:

```bash
pip install modal
modal token new
```

2. Create a Modal secret named `backend-secret` containing at least:

- `MODAL_SHARED_SECRET` (same value you set on the backend)
- `CLOUD_RUN_BASE_URL` (the backend’s public base URL, e.g. `https://backend-...run.app`)

3. Deploy:

```bash
python video_processing/deploy.py
```

4. From the Modal output, copy the trigger web endpoint URL and set it on the backend as:

- `MODAL_TRIGGER_BASE_URL=<that-url>`

---

## Deploy backend (Cloud Run)

1. Ensure the required Firebase/GCP APIs are enabled for your project:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

2. Deploy from repo root:

```bash
cd backend
gcloud run deploy backend --source . --region europe-west3 --allow-unauthenticated
```

3. Configure the backend’s environment variables in Cloud Run (see “Required configuration”).

4. Verify:

- `GET /` returns `{ "ok": true }`
- jobs endpoints require Firebase Auth

---

## Deploy frontend (Firebase Hosting)

1. Configure `frontend/.env.production` with the deployed backend URL:

- `VITE_BACKEND_URL=https://<cloud-run-url>`

2. Build and deploy:

```bash
cd frontend
npm install
npm run build
firebase deploy --only hosting
```

---

## End-to-end smoke test

1. Open the deployed frontend.
2. Sign in (Firebase Auth).
3. Select an ingress folder.
4. Drop a short `.mp4` file into it and confirm:
   - the upload progresses,
   - the job transitions through states,
   - the player opens once status is `succeeded`.
