#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.docs_agent.dotenv import load_repo_env  # noqa: E402
from tools.docs_agent.cache import LlmCache  # noqa: E402
from tools.docs_agent.lib import (  # noqa: E402
    DEFAULT_CODE_EXTS,
    DEFAULT_EXCLUDE_DIRS,
    folder_content_hash,
    iter_folders,
    load_state,
    norm_rel,
    repo_root,
    save_state,
    sha256_file,
    touch_state_for_scan,
    tracked_code_files,
)
from tools.docs_agent.llm_gemini import (  # noqa: E402
    GeminiError,
    config_from_env as gemini_config_from_env,
    gemini_chat_text_with_retries,
)
from tools.docs_agent.llm_openai import OpenAIError, config_from_env as openai_config_from_env, openai_respond_text  # noqa: E402
from tools.docs_agent.verify import VerificationError, verify_python_docs_only  # noqa: E402


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)\n```", re.DOTALL)

# Optional dependency: progress bars.
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

def _extract_first_fenced_block(text: str) -> str | None:
    m = _FENCE_RE.search(text)
    if not m:
        return None
    return m.group(1)


def _file_update_prompt(path: str, lang: str, src: str) -> tuple[str, str]:
    system = (
        "You update developer documentation in source files.\n"
        "Rules:\n"
        "- ONLY add/update comments/docstrings/JSDoc; do not change runtime behavior.\n"
        "- Do not rename symbols, reorder imports, or change formatting beyond what is needed for docs.\n"
        "- If docs are already good, return the file unchanged.\n"
        "- Output ONLY the full updated file content, inside one fenced code block.\n"
    )
    user = (
        f"File path: {path}\n"
        f"Language: {lang}\n\n"
        "Update documentation for the entire file (module/file header, classes, functions).\n"
        "Keep it concise and accurate.\n\n"
        "Current file content:\n"
        f"```{lang}\n{src}\n```\n"
    )
    return system, user


def _readme_prompt(folder_rel: str, files_rel: list[str], existing: str | None) -> tuple[str, str]:
    system = (
        "You write short, practical folder README.md documentation.\n"
        "Rules:\n"
        "- Keep it concise; prefer bullets.\n"
        "- Don't invent features; if uncertain, include a TODO.\n"
        "- Output ONLY markdown content (no code fences).\n"
    )
    file_list = "\n".join(f"- {p}" for p in files_rel[:200])
    user = f"Folder: {folder_rel}\n\nTracked code files in this folder (subset):\n{file_list}\n\n"
    if existing:
        user += f"Existing README.md:\n\n{existing}\n\n"
    user += "Write/update README.md for this folder."
    return system, user


def _llm_text(provider: str, *, system: str, user: str) -> str:
    if provider == "gemini":
        return gemini_chat_text_with_retries(gemini_config_from_env(), system=system, user=user)
    if provider == "openai":
        # Map system->instructions (close enough for our docs use-case).
        return openai_respond_text(openai_config_from_env(), instructions=system, user_input=user)
    raise ValueError(f"Unknown provider: {provider}")


def _ensure_folder_readme_stub(folder: Path) -> str:
    name = folder.name
    return (
        f"# {name}\n\n"
        "## Overview\n\n"
        "- TODO: Describe what this folder contains.\n\n"
        "## Key Files\n\n"
        "- TODO\n"
    )


def _strip_trailing_whitespace(text: str) -> str:
    has_trailing_newline = text.endswith("\n")
    cleaned = "\n".join(line.rstrip() for line in text.splitlines())
    return cleaned + ("\n" if has_trailing_newline else "")


def _progress(iterable, *, total: int | None, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="docs-agent-generate", description="Generate docs only for changed files/folders.")
    parser.add_argument("--apply", action="store_true", help="Write changes to disk. Default is dry-run.")
    parser.add_argument("--force", action="store_true", help="Ignore cached hashes and process everything (useful for the first run).")
    parser.add_argument(
        "--llm-sleep-seconds",
        type=float,
        default=float(os.environ.get("DOCS_AGENT_LLM_SLEEP_SECONDS", "0") or 0),
        help="Sleep between LLM requests to reduce 429s (or set DOCS_AGENT_LLM_SLEEP_SECONDS).",
    )
    parser.add_argument("--max-files", type=int, default=25, help="Max code files to process per run.")
    parser.add_argument("--max-folders", type=int, default=25, help="Max folders to process per run.")
    parser.add_argument("--include-ext", action="append", default=[], help="Additional file extension to include.")
    parser.add_argument("--exclude-dir", action="append", default=[], help="Directory name to exclude.")
    parser.add_argument("--update-inline-docs", action="store_true", help="Update docstrings/JSDoc inside code files.")
    parser.add_argument("--update-readmes", action="store_true", help="Create/update per-folder README.md files.")
    parser.add_argument(
        "--llm",
        choices=["gemini", "openai", "none"],
        default="none",
        help="LLM provider to use. 'none' only creates README stubs for missing readmes.",
    )
    parser.add_argument("--write-state", action="store_true", help="Update cache state after a successful apply.")
    parser.add_argument("--format-python", action="store_true", help="Run `ruff format` on updated Python files (if ruff is installed).")
    args = parser.parse_args(argv)

    if not args.update_inline_docs and not args.update_readmes:
        print("Nothing to do: set --update-inline-docs and/or --update-readmes.", file=sys.stderr)
        return 2

    root = repo_root(Path.cwd())
    load_repo_env(root)
    llm_cache = LlmCache.from_env(root)

    extra_exts = {e if e.startswith(".") else f".{e}" for e in args.include_ext}
    exts = {*(DEFAULT_CODE_EXTS | extra_exts)}
    exclude_dirs = {*(DEFAULT_EXCLUDE_DIRS | set(args.exclude_dir))}

    prompt_version = "docs_agent/v1"
    state = load_state(root, prompt_version=prompt_version)
    cache_invalidated = args.force or (state.prompt_version != prompt_version)

    files = tracked_code_files(root, exts=exts)
    file_hashes: dict[str, str] = {}
    changed_files: list[Path] = []
    for p in files:
        rel = norm_rel(root, p)
        try:
            h = sha256_file(p)
        except Exception:
            continue
        file_hashes[rel] = h
        prev = state.file_hashes.get(rel)
        if cache_invalidated or prev != h:
            changed_files.append(p)

    updated_python_files: list[Path] = []
    if args.update_inline_docs and args.llm in {"gemini", "openai"}:
        processed = 0
        inline_targets = changed_files[: args.max_files]
        for p in _progress(inline_targets, total=len(inline_targets), desc="Inline docs"):
            if processed >= args.max_files:
                break
            rel = norm_rel(root, p)
            lang = "python" if p.suffix.lower() == ".py" else "tsx" if p.suffix.lower() == ".tsx" else "ts"
            src = p.read_text(encoding="utf-8")
            system, user = _file_update_prompt(rel, lang, src)
            try:
                if args.llm == "gemini":
                    cfg = gemini_config_from_env()
                    key = llm_cache.build_key(
                        provider="gemini",
                        model=cfg.model,
                        base_url=cfg.base_url,
                        system=system,
                        user=user,
                        prompt_version=prompt_version,
                    )
                    text, _ = llm_cache.get_or_set(
                        key=key,
                        provider="gemini",
                        model=cfg.model,
                        base_url=cfg.base_url,
                        prompt_version=prompt_version,
                        system=system,
                        user=user,
                        fetch=lambda: gemini_chat_text_with_retries(cfg, system=system, user=user),
                    )
                else:
                    cfg = openai_config_from_env()
                    key = llm_cache.build_key(
                        provider="openai",
                        model=cfg.model,
                        base_url=cfg.base_url,
                        system=system,
                        user=user,
                        prompt_version=prompt_version,
                    )
                    text, _ = llm_cache.get_or_set(
                        key=key,
                        provider="openai",
                        model=cfg.model,
                        base_url=cfg.base_url,
                        prompt_version=prompt_version,
                        system=system,
                        user=user,
                        fetch=lambda: openai_respond_text(cfg, instructions=system, user_input=user),
                    )
            except (OpenAIError, GeminiError) as e:
                print(f"[llm error] {rel}: {e}", file=sys.stderr)
                if args.llm_sleep_seconds > 0:
                    time.sleep(args.llm_sleep_seconds)
                continue

            updated = _strip_trailing_whitespace(_extract_first_fenced_block(text) or text)
            if p.suffix.lower() == ".py":
                try:
                    verify_python_docs_only(src, updated)
                except VerificationError as e:
                    print(f"[verify failed] {rel}: {e}", file=sys.stderr)
                    if args.llm_sleep_seconds > 0:
                        time.sleep(args.llm_sleep_seconds)
                    continue

            if updated != src:
                processed += 1
                print(f"[file] {rel} {'APPLY' if args.apply else 'DRY'}")
                if args.apply:
                    p.write_text(updated, encoding="utf-8")
                    if p.suffix.lower() == ".py":
                        updated_python_files.append(p)
            if args.llm_sleep_seconds > 0:
                time.sleep(args.llm_sleep_seconds)

        if args.apply and args.format_python and updated_python_files:
            ruff = shutil.which("ruff")
            if not ruff:
                print("[format] ruff not found; skipping python formatting.", file=sys.stderr)
            else:
                proc = subprocess.run(
                    [ruff, "format", *[str(p) for p in updated_python_files]],
                    cwd=str(root),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                sys.stdout.write(proc.stdout)
                sys.stderr.write(proc.stderr)
                if proc.returncode != 0:
                    print("[format] ruff format failed.", file=sys.stderr)

    if args.update_readmes:
        processed_folders = 0
        current_files = tracked_code_files(root, exts=exts)
        current_file_hashes = {norm_rel(root, p): sha256_file(p) for p in current_files}
        # Precompute folder candidates so we can show progress.
        folder_candidates: list[Path] = []
        for folder in iter_folders(current_files, root=root, exclude_dirs=exclude_dirs):
            if len(folder_candidates) >= args.max_folders:
                break
            rel_folder = norm_rel(root, folder)
            folder_h = folder_content_hash(root, folder, file_hashes=current_file_hashes)
            prev_h = state.folder_hashes.get(rel_folder)
            readme_path = folder / "README.md"
            existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
            if existing is not None and (not cache_invalidated and prev_h == folder_h):
                continue
            if args.llm == "none" and existing is not None:
                continue
            folder_candidates.append(folder)

        for folder in _progress(folder_candidates, total=len(folder_candidates), desc="Folder READMEs"):
            if processed_folders >= args.max_folders:
                break
            rel_folder = norm_rel(root, folder)
            folder_h = folder_content_hash(root, folder, file_hashes=current_file_hashes)
            prev_h = state.folder_hashes.get(rel_folder)

            readme_path = folder / "README.md"
            existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
            if existing is not None and (not cache_invalidated and prev_h == folder_h):
                continue

            files_rel = sorted(
                [norm_rel(root, p) for p in current_files if norm_rel(root, p).startswith(rel_folder.rstrip("/") + "/")]
            )

            if args.llm == "none":
                if existing is not None:
                    continue
                new_readme = _ensure_folder_readme_stub(folder)
            else:
                system, user = _readme_prompt(rel_folder, files_rel, existing)
                try:
                    if args.llm == "gemini":
                        cfg = gemini_config_from_env()
                        key = llm_cache.build_key(
                            provider="gemini",
                            model=cfg.model,
                            base_url=cfg.base_url,
                            system=system,
                            user=user,
                            prompt_version=prompt_version,
                        )
                        text, _ = llm_cache.get_or_set(
                            key=key,
                            provider="gemini",
                            model=cfg.model,
                            base_url=cfg.base_url,
                            prompt_version=prompt_version,
                            system=system,
                            user=user,
                            fetch=lambda: gemini_chat_text_with_retries(cfg, system=system, user=user),
                        )
                    else:
                        cfg = openai_config_from_env()
                        key = llm_cache.build_key(
                            provider="openai",
                            model=cfg.model,
                            base_url=cfg.base_url,
                            system=system,
                            user=user,
                            prompt_version=prompt_version,
                        )
                        text, _ = llm_cache.get_or_set(
                            key=key,
                            provider="openai",
                            model=cfg.model,
                            base_url=cfg.base_url,
                            prompt_version=prompt_version,
                            system=system,
                            user=user,
                            fetch=lambda: openai_respond_text(cfg, instructions=system, user_input=user),
                        )
                    new_readme = _strip_trailing_whitespace(text.strip() + "\n")
                except (OpenAIError, GeminiError) as e:
                    print(f"[llm error] {rel_folder}/README.md: {e}", file=sys.stderr)
                    if args.llm_sleep_seconds > 0:
                        time.sleep(args.llm_sleep_seconds)
                    continue
                if args.llm_sleep_seconds > 0:
                    time.sleep(args.llm_sleep_seconds)

            if (existing or "") != new_readme:
                processed_folders += 1
                print(f"[readme] {rel_folder}/README.md {'APPLY' if args.apply else 'DRY'}")
                if args.apply:
                    readme_path.write_text(new_readme, encoding="utf-8")

    if args.apply and args.write_state:
        refreshed_files = tracked_code_files(root, exts=exts)
        refreshed_file_hashes = {norm_rel(root, p): sha256_file(p) for p in refreshed_files}
        refreshed_folder_hashes: dict[str, str] = {}
        for folder in iter_folders(refreshed_files, root=root, exclude_dirs=exclude_dirs):
            rel_folder = norm_rel(root, folder)
            refreshed_folder_hashes[rel_folder] = folder_content_hash(root, folder, file_hashes=refreshed_file_hashes)
        save_state(
            root,
            touch_state_for_scan(
                state,
                file_hashes=refreshed_file_hashes,
                folder_hashes=refreshed_folder_hashes,
                prompt_version=prompt_version,
            ),
        )

    if not args.apply:
        print("Dry run complete. Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
