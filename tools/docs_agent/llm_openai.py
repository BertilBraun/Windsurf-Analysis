from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 120


class OpenAIError(RuntimeError):
    pass


def _extract_text_from_responses_api(resp: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in resp.get("output") or []:
        if item.get("type") != "message":
            continue
        if item.get("role") != "assistant":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                chunks.append(c["text"])
    text = "\n".join(chunks).strip()
    if text:
        return text
    if isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
        return resp["output_text"].strip()
    raise OpenAIError("No assistant text found in response.")


def openai_respond_text(config: OpenAIConfig, *, instructions: str, user_input: str) -> str:
    url = f"{config.base_url.rstrip('/')}/responses"
    payload = {
        "model": config.model,
        "instructions": instructions,
        "input": user_input,
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
        raise OpenAIError(f"OpenAI request failed: {e}") from e

    try:
        resp = json.loads(body)
    except Exception as e:
        raise OpenAIError(f"OpenAI response was not JSON: {e}") from e

    if "error" in resp:
        raise OpenAIError(str(resp["error"]))
    return _extract_text_from_responses_api(resp)


def config_from_env() -> OpenAIConfig:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise OpenAIError("OPENAI_API_KEY is not set.")
    model = os.environ.get("DOCS_AGENT_OPENAI_MODEL", "").strip() or os.environ.get("OPENAI_MODEL", "").strip()
    if not model:
        raise OpenAIError("Set DOCS_AGENT_OPENAI_MODEL (or OPENAI_MODEL) to choose a model.")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
    return OpenAIConfig(api_key=api_key, model=model, base_url=base_url)

