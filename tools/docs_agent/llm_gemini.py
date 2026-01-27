from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    model: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    timeout_seconds: int = 120


class GeminiError(RuntimeError):
    pass


def config_from_env() -> GeminiConfig:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not set.")
    model = os.environ.get("DOCS_AGENT_GEMINI_MODEL", "").strip() or "gemini-3-flash-preview"
    base_url = os.environ.get("DOCS_AGENT_GEMINI_BASE_URL", "").strip() or "https://generativelanguage.googleapis.com/v1beta/openai"
    return GeminiConfig(api_key=api_key, model=model, base_url=base_url)


def _extract_text(resp: dict[str, Any]) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        raise GeminiError(f"Unexpected response format: {e}") from e


def gemini_chat_text(config: GeminiConfig, *, system: str, user: str) -> str:
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        method="POST",
        data=data,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_seconds) as f:
            body = f.read().decode("utf-8")
    except Exception as e:
        raise GeminiError(f"Gemini request failed: {e}") from e

    try:
        resp = json.loads(body)
    except Exception as e:
        raise GeminiError(f"Gemini response was not JSON: {e}") from e

    if "error" in resp:
        raise GeminiError(str(resp["error"]))
    return _extract_text(resp)

