#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.docs_agent.lib import repo_root, run_git, state_path  # noqa: E402


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _ensure_clean_worktree(root: Path, wt_path: Path, ref: str) -> None:
    if wt_path.exists():
        _run(['git', 'worktree', 'remove', '--force', str(wt_path)], cwd=root)
    proc = _run(['git', 'worktree', 'add', '--force', '--detach', str(wt_path), ref], cwd=root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or 'git worktree add failed')


def main(argv: list[str]) -> int:
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

    env = os.environ.copy()
    env['DOCS_AGENT_STATE_PATH'] = str(state_path(root).resolve())
    # Worktrees don't include untracked `.env` files, so point the generator at the real env file.
    env.setdefault('DOCS_AGENT_ENV_PATH', str((root / '.env').resolve()))

    generate_py = wt_path / 'tools' / 'docs_agent' / 'generate.py'
    proc = _run([sys.executable, str(generate_py)] + [a for a in args.generate_args if a != '--'], cwd=wt_path, env=env)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        return proc.returncode

    diff = run_git(['diff'], cwd=wt_path)
    if not diff.strip():
        print('No changes produced.')
        return 0

    patch_path.write_text(diff, encoding='utf-8')
    print(f'Wrote patch: {patch_path}')

    if not args.no_apply:
        apply_proc = _run(['git', 'apply', str(patch_path)], cwd=root)
        if apply_proc.returncode != 0:
            sys.stderr.write(apply_proc.stderr)
            sys.stderr.write(apply_proc.stdout)
            return apply_proc.returncode
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
