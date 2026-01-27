from __future__ import annotations

import os
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip("'").strip('"')
    if not key:
        return None
    return key, value


def load_dotenv(path: Path) -> None:
    """
    Loads KEY=VALUE lines from a `.env` file into process environment.
    - Does not overwrite existing env vars.
    - Supports comments and simple quoted values.
    """
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return
    for raw in text.splitlines():
        parsed = _parse_env_line(raw)
        if not parsed:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def load_repo_env(repo_root: "Path") -> Path | None:
    """
    Loads env vars from:
    - `DOCS_AGENT_ENV_PATH` if set, else
    - `<repo_root>/.env`
    Returns the path used (or None if no file existed).
    """
    override = os.environ.get("DOCS_AGENT_ENV_PATH", "").strip()
    env_path = Path(override) if override else (repo_root / ".env")
    if env_path.exists():
        load_dotenv(env_path)
        return env_path
    return None

