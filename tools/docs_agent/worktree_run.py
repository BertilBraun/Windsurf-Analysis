#!/usr/bin/env python3
"""Runs documentation generation in a temporary Git worktree.

Creates a clean worktree, mirrors local changes, runs the generator, and
applies the resulting changes back to the main repository.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.docs_agent.lib import repo_root, run_git, state_path  # noqa: E402


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _run_stream(cmd: list[str], *, cwd: Path, env: dict[str, str] | None) -> int:
    """
    Run a process and stream its stdout/stderr directly to our stdout/stderr.

    Uses byte streaming to preserve progress bars (tqdm carriage returns) and avoid
    Windows encoding decode errors.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    def pump(src, dst):
        while True:
            chunk = src.read(4096)
            if not chunk:
                break
            dst.write(chunk)
            dst.flush()

    t_out = threading.Thread(target=pump, args=(proc.stdout, sys.stdout.buffer), daemon=True)
    t_err = threading.Thread(target=pump, args=(proc.stderr, sys.stderr.buffer), daemon=True)
    t_out.start()
    t_err.start()
    rc = proc.wait()
    t_out.join(timeout=5)
    t_err.join(timeout=5)
    return rc


def _ensure_clean_worktree(root: Path, wt_path: Path, ref: str) -> None:
    if wt_path.exists():
        _run(['git', 'worktree', 'remove', '--force', str(wt_path)], cwd=root)
    proc = _run(['git', 'worktree', 'add', '--force', '--detach', str(wt_path), ref], cwd=root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or 'git worktree add failed')


def _apply_worktree_changes_by_copy(*, root: Path, wt_path: Path) -> tuple[int, int]:
    """
    Apply worktree changes back to the main repo by copying files.

    This avoids `git apply` failures on Windows (CRLF/LF, missing final newline, etc.)
    and prevents .rej files when patches don't match.
    """
    name_status = run_git(['diff', '--name-status', '--no-renames'], cwd=wt_path)
    copied = 0
    deleted = 0
    for line in name_status.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0].strip()
        if len(parts) < 2:
            continue
        rel = parts[1].strip()
        if not rel or rel.startswith(".git/") or rel.startswith(".docs_agent/"):
            continue
        src = wt_path / rel
        dst = root / rel

        if status.startswith("D"):
            if dst.exists():
                dst.unlink()
                deleted += 1
            continue

        if not src.exists() or src.is_dir():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        copied += 1
    return copied, deleted


def _sync_root_changes_into_worktree(root: Path, wt_path: Path) -> None:
    """
    Mirror the current working-tree contents of modified tracked files into the worktree.

    Why: if your main working directory is dirty, we want generation to run against that
    exact state so the produced patch applies cleanly back on top of it.
    """
    status = run_git(['status', '--porcelain=v1'], cwd=root)
    if not status.strip():
        return

    for raw in status.splitlines():
        if not raw:
            continue
        if raw.startswith('??'):
            path_part = raw[3:].strip()
            rel_path = Path(path_part)
            # Never copy secrets or agent artifacts.
            if rel_path.as_posix().startswith('.docs_agent/') or rel_path.name == '.env':
                continue
            if rel_path.suffix.lower() == ".rej":
                continue
            src = root / rel_path
            dst = wt_path / rel_path
            if not src.exists() or src.is_dir():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            continue
        xy = raw[:2]
        path_part = raw[3:].strip()
        if ' -> ' in path_part:
            continue
        rel_path = Path(path_part)
        src = root / rel_path
        dst = wt_path / rel_path

        if 'D' in xy:
            if dst.exists():
                dst.unlink()
            continue

        if not src.exists() or src.is_dir():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


def main(argv: list[str]) -> int:
    """Main entry point for the docs-agent-worktree tool."""
    parser = argparse.ArgumentParser(
        prog='docs-agent-worktree',
        description='Run docs generation in a clean detached worktree and apply the patch to the current repo.',
    )
    parser.add_argument('--ref', default='HEAD', help='Git ref to use for the clean worktree (default: HEAD).')
    parser.add_argument(
        '--worktree-path', default='', help='Override worktree path (default: <repo>/.docs_agent/worktree).'
    )
    parser.add_argument(
        '--keep-worktree', action='store_true', help='Do not remove the worktree folder after generating.'
    )
    parser.add_argument('--no-apply', action='store_true', help='Do not apply the generated patch to the current repo.')
    parser.add_argument(
        '--patch-path', default='', help='Where to write the patch (default: <repo>/.docs_agent/last.patch).'
    )
    parser.add_argument('generate_args', nargs=argparse.REMAINDER, help='Args passed through to generate.py.')
    args = parser.parse_args(argv)

    root = repo_root(Path.cwd())
    wt_path = Path(args.worktree_path).resolve() if args.worktree_path else (root / '.docs_agent' / 'worktree')
    patch_path = Path(args.patch_path).resolve() if args.patch_path else (root / '.docs_agent' / 'last.patch')
    patch_path.parent.mkdir(parents=True, exist_ok=True)

    _ensure_clean_worktree(root, wt_path, ref=args.ref)
    _sync_root_changes_into_worktree(root, wt_path)

    # Stage the synced baseline so the diff only contains generator-produced changes.
    stage_proc = _run(['git', 'add', '-A'], cwd=wt_path)
    if stage_proc.returncode != 0:
        sys.stderr.write(stage_proc.stderr)
        sys.stderr.write(stage_proc.stdout)
        return stage_proc.returncode

    env = os.environ.copy()
    env['DOCS_AGENT_STATE_PATH'] = str(state_path(root).resolve())
    env.setdefault('DOCS_AGENT_CACHE_DIR', str((root / '.docs_agent' / 'llm_cache').resolve()))
    # Worktrees don't include untracked `.env` files, so point the generator at the real env file.
    env.setdefault('DOCS_AGENT_ENV_PATH', str((root / '.env').resolve()))
    # Make the child process write UTF-8 so our decoding is stable.
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    env.setdefault('PYTHONUTF8', '1')

    generate_py = wt_path / 'tools' / 'docs_agent' / 'generate.py'
    rc = _run_stream(
        [sys.executable, str(generate_py)] + [a for a in args.generate_args if a != '--'],
        cwd=wt_path,
        env=env,
    )
    if rc != 0:
        return rc

    # Ensure newly-created untracked files show up in the diff (e.g. new README.md files).
    wt_status = run_git(['status', '--porcelain=v1'], cwd=wt_path)
    untracked: list[str] = []
    for line in wt_status.splitlines():
        if line.startswith('?? '):
            untracked.append(line[3:].strip())
    if untracked:
        addn_proc = _run(['git', 'add', '-N', '--', *untracked], cwd=wt_path)
        if addn_proc.returncode != 0:
            sys.stderr.write(addn_proc.stderr)
            sys.stderr.write(addn_proc.stdout)
            return addn_proc.returncode

    diff = run_git(['diff'], cwd=wt_path)
    if not (diff or "").strip():
        print('No changes produced.')
        return 0

    patch_path.write_text(diff, encoding='utf-8')
    print(f'Wrote patch: {patch_path}')

    if not args.no_apply:
        copied, deleted = _apply_worktree_changes_by_copy(root=root, wt_path=wt_path)
        print(f'Applied changes: {copied} file(s) updated, {deleted} deleted.')
        state_proc = _run(
            [sys.executable, str(root / 'tools' / 'docs_agent' / 'run.py'), '--write-state'], cwd=root, env=env
        )
        sys.stdout.write(state_proc.stdout)
        sys.stderr.write(state_proc.stderr)
        if state_proc.returncode != 0:
            return state_proc.returncode

    if not args.keep_worktree:
        _run(['git', 'worktree', 'remove', '--force', str(wt_path)], cwd=root)

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
