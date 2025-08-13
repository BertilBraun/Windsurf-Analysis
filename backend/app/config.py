import os


def get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f'Missing required environment variable: {name}')
    return value


class Settings:
    # Server
    APP_NAME: str = os.getenv('APP_NAME', 'windsurf-analysis-api')
    APP_ENV: str = os.getenv('APP_ENV', 'dev')
    CORS_ORIGINS: str = os.getenv('CORS_ORIGINS', '*')

    # Auth
    BASIC_AUTH_REALM: str = os.getenv('BASIC_AUTH_REALM', 'Windsurf Analysis')
    # Comma-separated bootstrap users: "user1:hashed, user2:hashed"; optional if using DB-seeded users
    BOOTSTRAP_USERS: str | None = os.getenv('BOOTSTRAP_USERS')

    # Database (Neon Postgres)
    DATABASE_URL: str = get_env('DATABASE_URL', None)

    # Object storage (R2/S3 compatible)
    S3_ENDPOINT_URL: str | None = os.getenv('S3_ENDPOINT_URL')  # e.g., https://<accountid>.r2.cloudflarestorage.com
    S3_REGION: str = os.getenv('S3_REGION', 'auto')
    S3_BUCKET: str = get_env('S3_BUCKET', None)
    S3_ACCESS_KEY_ID: str = get_env('S3_ACCESS_KEY_ID', None)
    S3_SECRET_ACCESS_KEY: str = get_env('S3_SECRET_ACCESS_KEY', None)

    # Storage layout
    PREFIX_AC_VIDEOS: str = os.getenv('PREFIX_AC_VIDEOS', 'ac-videos/')
    PREFIX_RESULTS_JSON: str = os.getenv('PREFIX_RESULTS_JSON', 'results-json/')

    # Quotas
    MAX_JOBS_PER_USER: int = int(os.getenv('MAX_JOBS_PER_USER', '5'))

    # Modal / inference
    MODAL_INVOKE_URL: str = get_env('MODAL_INVOKE_URL', None)  # HTTPS URL of Modal web endpoint
    BACKEND_WEBHOOK_SECRET: str = get_env('BACKEND_WEBHOOK_SECRET', None)
    BACKEND_PUBLIC_BASE_URL: str = get_env('BACKEND_PUBLIC_BASE_URL', None)  # e.g., https://api.example.com

    # Signed URL expirations (seconds)
    SIGNED_URL_TTL: int = int(os.getenv('SIGNED_URL_TTL', '900'))

    # Admin secret for creating users via API
    USER_CREATE_SECRET: str = get_env('USER_CREATE_SECRET', None)


settings = Settings()
