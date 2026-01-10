import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    # CORS
    allowed_origins: tuple[str, ...] = (
        # Firebase Hosting (prod)
        'https://gybelock-00.web.app',
        'https://gybelock.de',
        # Local dev
        'http://localhost',
        'http://localhost:3000',
        'http://localhost:5173',
        'http://127.0.0.1',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:5173',
    )

    allow_origin_regex: str = r'^http://(\[::1\]|localhost|127\.0\.0\.1)(:\\d+)?$'

    # Firestore
    firestore_database: str = os.getenv('FIRESTORE_DATABASE', '(default)')

    # Quotas
    max_jobs_per_user_default: int = int(os.getenv('MAX_JOBS_PER_USER', '5'))

    # Internal (Modal -> Cloud Run) auth
    modal_secret_header: str = 'X-Modal-Secret'
    modal_shared_secret: str | None = os.getenv('MODAL_SHARED_SECRET')

    # Storage
    firebase_storage_bucket: str | None = os.getenv('FIREBASE_STORAGE_BUCKET')

    # Modal (Cloud Run -> Modal) trigger endpoint
    modal_trigger_base_url: str = os.getenv('MODAL_TRIGGER_BASE_URL', '').rstrip('/')


settings = Settings()
