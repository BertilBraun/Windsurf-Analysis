from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(data: dict) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    key: str


class LlmCache:
    def __init__(self, cache_dir: Path):
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, repo_root: Path) -> "LlmCache":
        override = os.environ.get("DOCS_AGENT_CACHE_DIR", "").strip()
        cache_dir = Path(override) if override else (repo_root / ".docs_agent" / "llm_cache")
        return cls(cache_dir=cache_dir)

    def _path_for_key(self, key: CacheKey) -> Path:
        return self._cache_dir / f"{key.key}.json"

    def build_key(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        system: str,
        user: str,
        prompt_version: str,
    ) -> CacheKey:
        key_material = {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "prompt_version": prompt_version,
            "system_sha256": _sha256_text(system),
            "user_sha256": _sha256_text(user),
        }
        return CacheKey(_sha256_json(key_material))

    def get(self, key: CacheKey) -> str | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        text = raw.get("response_text")
        return text if isinstance(text, str) else None

    def set(
        self,
        key: CacheKey,
        *,
        provider: str,
        model: str,
        base_url: str,
        prompt_version: str,
        system: str,
        user: str,
        response_text: str,
    ) -> None:
        path = self._path_for_key(key)
        tmp = path.with_suffix(".tmp")
        record = {
            "version": 1,
            "created_at_unix": int(time.time()),
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "prompt_version": prompt_version,
            "system_sha256": _sha256_text(system),
            "user_sha256": _sha256_text(user),
            "response_text": response_text,
        }
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    def get_or_set(
        self,
        *,
        key: CacheKey,
        provider: str,
        model: str,
        base_url: str,
        prompt_version: str,
        system: str,
        user: str,
        fetch: Callable[[], str],
    ) -> tuple[str, bool]:
        cached = self.get(key)
        if cached is not None:
            return cached, True
        text = fetch()
        self.set(
            key,
            provider=provider,
            model=model,
            base_url=base_url,
            prompt_version=prompt_version,
            system=system,
            user=user,
            response_text=text,
        )
        return text, False

