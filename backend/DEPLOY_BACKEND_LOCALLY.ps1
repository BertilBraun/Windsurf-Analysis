$env:DATABASE_URL="postgresql+asyncpg://USER:PASSWORD@HOST/DB"
$env:S3_ENDPOINT_URL="https://<account-id>.r2.cloudflarestorage.com"
$env:S3_REGION="auto"
$env:S3_BUCKET="windsurf-analysis"
$env:S3_ACCESS_KEY_ID="..."
$env:S3_SECRET_ACCESS_KEY="..."
$env:BACKEND_WEBHOOK_SECRET="replace-with-random"
$env:BACKEND_PUBLIC_BASE_URL="http://localhost:8000"
$env:MODAL_INVOKE_URL="https://modal-web-endpoint"
$env:CORS_ORIGINS="http://localhost:5173"

uvicorn app.main:app --reload --port 8000
