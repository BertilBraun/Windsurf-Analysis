from __future__ import annotations

# Thin entrypoint so `modal serve main.py` from repo root works reliably.
# It re-exports the Modal app/function defined in `modal_app/main.py`.

from modal_app.main import app, fastapi_app  # noqa: F401
