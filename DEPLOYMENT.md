Infrastructure setup guide for Windsurf Analysis (Modal + FastAPI + Neon + R2)

Prerequisites

- Python 3.11+
- Node 18+
- Accounts: Neon (Postgres), Cloudflare R2 (S3), Modal.com, Vercel/Render (for FastAPI/React)

Environment variables (shared)

- DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST/DB
- S3_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
- S3_REGION=auto
- S3_BUCKET=<bucket-name>
- S3_ACCESS_KEY_ID=...
- S3_SECRET_ACCESS_KEY=...
- BACKEND_WEBHOOK_SECRET=<random>
- BACKEND_PUBLIC_BASE_URL=https://api.example.com
- MODAL_INVOKE_URL=https://<modal-web-endpoint>
- CORS_ORIGINS=https://app.example.com
- USER_CREATE_SECRET=<admin-secret>

Database setup (Neon)

1. Create a new Neon project and DB.
2. Set DATABASE_URL to async URL (use postgresql+asyncpg scheme).
3. The backend will auto-create tables on startup.

Object storage (Cloudflare R2)

1. Create a bucket (e.g., windsurf-analysis).
2. Create an API token with read+write to the bucket.
3. Note Account ID for endpoint URL, set env vars above.

Modal deployment

1. pip install modal-client
2. modal token new
3. Deploy app:
   modal deploy modal_app/inference.py
4. Copy the printed web endpoint URL into MODAL_INVOKE_URL.
5. Note: Modal no longer needs storage credentials; the backend sends the AC bytes directly via multipart to Modal.

Backend deployment (Render or similar)

1. Create a Python service with Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
2. Set all env vars listed above.
3. Deploy.

Frontend deployment

1. Scaffold with Vite React TS (outside this repo scope). Configure .env with API base.
2. Implement flows per PRODUCTION_REQUIREMENTS.md.

