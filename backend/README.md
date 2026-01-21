## FastAPI backend (Cloud Run-ready)

### Run locally (Python)
From repo root:

```bash
python -m venv .venv
. .venv/bin/activate  # (Linux/Mac) on Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/main.py
```

### Run locally (Firebase Auth + Firestore enabled)
You need a **Firebase Admin SDK service account** for your Firebase project.

Then run (Windows PowerShell):

```powershell
gcloud auth application-default login
python backend/main.py
```

### Deploy to Google Cloud Run
From repo root (example):

```bash
cd backend
gcloud init
gcloud auth login
gcloud config set project gybelock-00
gcloud run deploy backend \
  --source . \
  --region europe-west3 \
  --allow-unauthenticated
```

### Environment variables
- `MODAL_SHARED_SECRET`: shared secret for Modal -> Cloud Run internal endpoints.
- `MODAL_TRIGGER_BASE_URL`: Modal trigger web endpoint base URL (used to start processing).
- `FIREBASE_STORAGE_BUCKET`: bucket name, e.g. `gybelock-00.appspot.com` (used to build `gs://...` URIs).

### Job results storage
Job results are written to Firebase Storage / GCS as JSON (instead of Firestore subcollections) to avoid Firestore document size limits.
By default the backend uses `gs://$FIREBASE_STORAGE_BUCKET/results/{job_id}.json`.

### CORS (calling from Firebase Hosting)
If your frontend is hosted on Firebase (e.g. `https://gybelock-00.web.app`) and you call this backend from the browser,
the backend must allow that origin via CORS. Configure `allowed_origins` in `main.py` and redeploy.

### Firebase Auth + Firestore
- **Auth**: the backend expects `Authorization: Bearer <Firebase ID token>` and verifies it (`GET /whoami`).
- **Firestore**: `POST /firestore/ping` writes a doc under `backendPings/{uid}`.
