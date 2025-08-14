from __future__ import annotations

# Thin entrypoint so `modal serve main.py` from repo root works reliably.
# It re-exports the Modal app/function defined in `server/main.py`.

from server.inference.inference import app, InferenceModel
