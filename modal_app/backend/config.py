import os
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()


def get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f'Missing required environment variable: {name}')
    return value


class Settings:
    # Server
    @classmethod
    @property
    def APP_NAME(cls) -> str:
        return os.getenv('APP_NAME', 'windsurf-analysis-backend')

    @classmethod
    @property
    def APP_ENV(cls) -> str:
        return os.getenv('APP_ENV', 'dev')

    # Database (Neon Postgres)
    @classmethod
    @property
    def DATABASE_URL(cls) -> str:
        return get_env('DATABASE_URL', None)

    # Object storage (R2/S3 compatible)
    @classmethod
    @property
    def S3_ENDPOINT_URL(cls) -> str | None:
        return os.getenv('S3_ENDPOINT_URL')  # e.g., https://<accountid>.r2.cloudflarestorage.com

    @classmethod
    @property
    def S3_REGION(cls) -> str:
        return os.getenv('S3_REGION', 'auto')

    @classmethod
    @property
    def S3_BUCKET(cls) -> str:
        return get_env('S3_BUCKET', None)

    @classmethod
    @property
    def S3_ACCESS_KEY_ID(cls) -> str:
        return get_env('S3_ACCESS_KEY_ID', None)

    @classmethod
    @property
    def S3_SECRET_ACCESS_KEY(cls) -> str:
        return get_env('S3_SECRET_ACCESS_KEY', None)

    # Storage layout
    @classmethod
    @property
    def PREFIX_AC_VIDEOS(cls) -> str:
        return os.getenv('PREFIX_AC_VIDEOS', 'ac-videos/')

    @classmethod
    @property
    def PREFIX_RESULTS_JSON(cls) -> str:
        return os.getenv('PREFIX_RESULTS_JSON', 'results-json/')

    # Quotas
    @classmethod
    @property
    def MAX_JOBS_PER_USER(cls) -> int:
        return int(os.getenv('MAX_JOBS_PER_USER', '5'))

    # Modal / inference
    @classmethod
    @property
    def BACKEND_WEBHOOK_SECRET(cls) -> str:
        return get_env('BACKEND_WEBHOOK_SECRET', None)

    @classmethod
    @property
    def BACKEND_PUBLIC_BASE_URL(cls) -> str:
        return get_env('BACKEND_PUBLIC_BASE_URL', None)  # e.g., https://api.example.com

    # Admin secret for creating users via API
    @classmethod
    @property
    def USER_CREATE_SECRET(cls) -> str:
        return get_env('USER_CREATE_SECRET', None)
