"""
Backend configuration module.

This module handles the loading of environment variables and defines the
Settings dataclass used for global application configuration.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Global application settings and environment configuration.

    Attributes:
        allowed_origins: Tuple of allowed CORS origins for web clients.
        allow_origin_regex: Regex for matching local development origins.
        firestore_database: Firestore database instance name.
        max_jobs_per_user_default: Default maximum jobs allowed per user.
        modal_secret_header: HTTP header name for Modal authentication.
        modal_shared_secret: Shared secret for Modal-to-Cloud Run authentication.
        firebase_storage_bucket: Firebase Storage bucket name.
        modal_trigger_base_url: Base URL for triggering Modal functions.
    """
    # CORS: Allowed origins for web clients
    allowed_origins: tuple[str, ...] = (
        # Firebase Hosting (prod)
        'https://gybelock-00.web.app',
        'https://gybelock.de',
        'https://gybelock.bertil-braun.de',
        # Local dev
        'http://localhost',
        'http://localhost:3000',
        'http://localhost:5173',
        'http://127.0.0.1',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:5173',
    )

    # Regex for matching local development origins
    allow_origin_regex: str = r'^http://(\[::1\]|localhost|127\.0\.0\.1)(:\\d+)?$'

    # Firestore: Database instance name
    firestore_database: str = os.getenv('FIRESTORE_DATABASE', '(default)')

    # Quotas: Default maximum jobs allowed per user
    max_jobs_per_user_default: int = int(os.getenv('MAX_JOBS_PER_USER', '5'))

    # Internal (Modal -> Cloud Run) authentication settings
    modal_secret_header: str = 'X-Modal-Secret'
    modal_shared_secret: str | None = os.getenv('MODAL_SHARED_SECRET')

    # Storage: Firebase Storage bucket name
    firebase_storage_bucket: str | None = os.getenv('FIREBASE_STORAGE_BUCKET')

    # Modal (Cloud Run -> Modal) trigger endpoint base URL
    modal_trigger_base_url: str = os.getenv('MODAL_TRIGGER_BASE_URL', '').rstrip('/')


# Global settings instance for application-wide access
settings = Settings()
