#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.docs_agent.dotenv import load_repo_env  # noqa: E402
from tools.docs_agent.lib import (  # noqa: E402
    DEFAULT_CODE_EXTS,
    DEFAULT_EXCLUDE_DIRS,
    folder_trigger_hash_with_readmes,
    iter_folders,
    load_state,
    norm_rel,
    repo_root,
    save_state,
    sha256_file,
    tracked_code_files,
    touch_state_for_scan,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="docs-agent", description="Hash-based inventory for doc updates (TS/TSX + Python).")
    parser.add_argument("--ext", action="append", default=[], help="Additional file extension to include (repeatable).")
    parser.add_argument("--exclude-dir", action="append", default=[], help="Directory name to exclude (repeatable).")
    parser.add_argument("--changed-only", action="store_true", help="Print only changed files (based on cached hashes).")
    parser.add_argument("--write-state", action="store_true", help="Write updated state after scanning.")
    parser.add_argument("--print-json", action="store_true", help="Output machine-readable JSON instead of human text.")
    args = parser.parse_args(argv)

    root = repo_root(Path.cwd())
    load_repo_env(root)

    extra_exts = {e if e.startswith(".") else f".{e}" for e in args.ext}
    exts = {*(DEFAULT_CODE_EXTS | extra_exts)}
    exclude_dirs = {*(DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir))}

    prompt_version = "docs_agent/v3"
    state = load_state(root, prompt_version=prompt_version)
    cache_invalidated = state.prompt_version != prompt_version

    files = tracked_code_files(root, exts=exts)
    file_hashes: dict[str, str] = {}
    changed: list[str] = []
    unchanged: list[str] = []

    for p in files:
        rel = norm_rel(root, p)
        try:
            h = sha256_file(p)
        except Exception:
            continue
        file_hashes[rel] = h
        prev = state.file_hashes.get(rel)
        if cache_invalidated or prev != h:
            changed.append(rel)
        else:
            unchanged.append(rel)

    folder_hashes: dict[str, str] = {}
    for folder in iter_folders(files, root=root, exclude_dirs=exclude_dirs):
        rel_folder = norm_rel(root, folder)
        folder_hashes[rel_folder] = folder_trigger_hash_with_readmes(root, folder, file_hashes=file_hashes, exclude_dirs=exclude_dirs)

    if args.print_json:
        print(
            json.dumps(
                {
                    "root": root.as_posix(),
                    "exts": sorted(exts),
                    "changed": sorted(changed),
                    "unchanged_count": len(unchanged),
                    "changed_count": len(changed),
                    "cache_invalidated": cache_invalidated,
                    "folder_count": len(folder_hashes),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"repo: {root}")
        print(f"tracked code files: {len(files)}")
        print(f"changed: {len(changed)} (cache invalidated: {cache_invalidated})")
        if args.changed_only:
            for rel in sorted(changed):
                print(rel)

    if args.write_state:
        save_state(
            root,
            touch_state_for_scan(state, file_hashes=file_hashes, folder_hashes=folder_hashes, prompt_version=prompt_version),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
