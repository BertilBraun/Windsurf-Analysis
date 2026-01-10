import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    cloud_run_base_url: str = os.getenv('CLOUD_RUN_BASE_URL', '').rstrip('/')

    modal_shared_secret: str = os.getenv('MODAL_SHARED_SECRET', '')


settings = Settings()
