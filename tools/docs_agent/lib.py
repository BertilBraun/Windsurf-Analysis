from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_CODE_EXTS = {".py", ".ts", ".tsx"}
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "tmp",
}


def run_git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout or ""


def repo_root(cwd: Path) -> Path:
    root = run_git(["rev-parse", "--show-toplevel"], cwd=cwd).strip()
    return Path(root)


def tracked_files(root: Path) -> list[Path]:
    out = run_git(["ls-files"], cwd=root)
    return [(root / line).resolve() for line in out.splitlines() if line.strip()]


def tracked_code_files(root: Path, exts: set[str]) -> list[Path]:
    return [p for p in tracked_files(root) if p.suffix.lower() in exts]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def norm_rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


@dataclasses.dataclass
class State:
    version: int
    prompt_version: str
    file_hashes: dict[str, str]
    folder_hashes: dict[str, str]
    updated_at_unix: int


def state_path(root: Path) -> Path:
    override = os.environ.get("DOCS_AGENT_STATE_PATH", "").strip()
    if override:
        return Path(override)
    return root / ".docs_agent" / "state.json"


def load_state(root: Path, prompt_version: str) -> State:
    p = state_path(root)
    if not p.exists():
        return State(version=1, prompt_version=prompt_version, file_hashes={}, folder_hashes={}, updated_at_unix=0)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if raw.get("version") != 1:
            raise ValueError("Unsupported state version")
        return State(
            version=1,
            prompt_version=str(raw.get("prompt_version") or ""),
            file_hashes=dict(raw.get("file_hashes") or {}),
            folder_hashes=dict(raw.get("folder_hashes") or {}),
            updated_at_unix=int(raw.get("updated_at_unix") or 0),
        )
    except Exception:
        return State(version=1, prompt_version=prompt_version, file_hashes={}, folder_hashes={}, updated_at_unix=0)


def save_state(root: Path, state: State) -> None:
    p = state_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "version": state.version,
                "prompt_version": state.prompt_version,
                "file_hashes": state.file_hashes,
                "folder_hashes": state.folder_hashes,
                "updated_at_unix": state.updated_at_unix,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def iter_folders(files: Iterable[Path], root: Path, exclude_dirs: set[str]) -> Iterator[Path]:
    seen: set[Path] = set()
    for f in files:
        folder = f.parent
        while True:
            if folder == root.parent:
                break
            try:
                rel = folder.resolve().relative_to(root.resolve())
            except Exception:
                break
            if not rel.parts:
                break
            if any(part in exclude_dirs for part in rel.parts):
                break
            if folder not in seen:
                seen.add(folder)
                yield folder
            if folder == root:
                break
            folder = folder.parent


def folder_content_hash(root: Path, folder: Path, file_hashes: dict[str, str]) -> str:
    rel_folder = norm_rel(root, folder)
    entries: list[tuple[str, str]] = []
    prefix = rel_folder.rstrip("/") + "/"
    for rel_path, h in file_hashes.items():
        if rel_path.startswith(prefix):
            entries.append((rel_path, h))
    entries.sort(key=lambda x: x[0])
    joined = "\n".join(f"{p}\t{h}" for p, h in entries).encode("utf-8")
    return sha256_bytes(joined)


def touch_state_for_scan(
    state: State, *, file_hashes: dict[str, str], folder_hashes: dict[str, str], prompt_version: str
) -> State:
    state.file_hashes = file_hashes
    state.folder_hashes = folder_hashes
    state.prompt_version = prompt_version
    state.updated_at_unix = int(time.time())
    return state
