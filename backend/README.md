## FastAPI backend (Cloud Run-ready)

### Run locally (Python)
From repo root:

```bash
python -m venv .venv
. .venv/bin/activate  # (Linux/Mac) on Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
python backend/main.py
```

### Deploy to Google Cloud Run
From repo root (example):

```bash
cd backend
gcloud init
gcloud auth login
gcloud config set project gybelock
gcloud run deploy backend \
  --source . \
  --region europe-west3 \
  --allow-unauthenticated
```
