#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ScanResult:
    locale_keys: set[str]
    used_keys: set[str]
    maybe_used: set[str]

    @property
    def unused_keys(self) -> list[str]:
        return sorted(self.locale_keys - self.used_keys)


def _flatten_locale_keys(obj: Any, prefix: str = '') -> Iterable[str]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                continue
            full_key = f'{prefix}.{key}' if prefix else key
            yield from _flatten_locale_keys(value, full_key)
        return

    if prefix:
        yield prefix


def _iter_files(root: Path, extensions: tuple[str, ...]) -> Iterable[Path]:
    extensions = tuple(ext.lower().lstrip('.') for ext in extensions)
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip('.') in extensions:
            yield path


def scan_used_keys(tsx_root: Path, locale_keys: set[str], extensions: tuple[str, ...]) -> ScanResult:
    used: set[str] = set()
    maybe_used: set[str] = set()
    for tsx_file in _iter_files(tsx_root, extensions=extensions):
        print(f'Scanning {tsx_file}...')
        try:
            text = tsx_file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = tsx_file.read_text(encoding='utf-8', errors='replace')

        for key in locale_keys:
            if key in text:
                used.add(key)
            elif key.rsplit('.', 1)[0] in text:
                path, last_part = key.rsplit('.', 1)
                if (path + '.${') in text or (path + ".'") in text or (path + '."') in text:
                    maybe_used.add(key)

    return ScanResult(locale_keys=locale_keys, used_keys=used, maybe_used=maybe_used)


def _delete_key_path(obj: Any, dotted_key: str) -> bool:
    parts = dotted_key.split('.')
    if not parts:
        return False

    cur = obj
    stack: list[tuple[dict[str, Any], str]] = []
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return False
        stack.append((cur, part))
        cur = cur[part]

    if not isinstance(cur, dict) or parts[-1] not in cur:
        return False

    del cur[parts[-1]]

    while stack:
        parent, key = stack.pop()
        child = parent.get(key)
        if isinstance(child, dict) and len(child) == 0:
            del parent[key]
        else:
            break

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Find unused i18n keys in a locale JSON file by scanning quoted string literals in TS/TSX.',
        epilog=(
            'Examples:\n'
            '  python scripts/i18n_unused_keys.py\n'
            '  python scripts/i18n_unused_keys.py --output tmp/unused_i18n_keys.txt\n'
            '  python scripts/i18n_unused_keys.py --ext tsx --ext ts\n'
            '  python scripts/i18n_unused_keys.py --remove\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--locale-file',
        type=Path,
        default=Path('../src/i18n/locales/en.json'),
        help='Path to the locale JSON file (default: frontend/src/i18n/locales/en.json).',
    )
    parser.add_argument(
        '--tsx-root',
        type=Path,
        default=Path('../src'),
        help='Directory to scan for TS/TSX usage (default: frontend/src).',
    )
    parser.add_argument(
        '--ext',
        action='append',
        default=['tsx'],
        help='File extension(s) to scan (default: --ext tsx). Can be repeated.',
    )
    parser.add_argument(
        '--ignore-key-regex',
        action='append',
        default=[],
        help='Regex for keys to ignore (treat as used). Can be repeated.',
    )
    parser.add_argument(
        '--maybe-output',
        type=Path,
        default=None,
        help=(
            "Write maybe-unused keys (one per line) to a file. A key is 'maybe-unused' if it's unused, "
            'but its prefix (everything except the last segment) appears as a string literal (or template static part).'
        ),
    )
    parser.add_argument(
        '--show-maybe',
        action='store_true',
        help='Also print maybe-unused keys to stdout (after unused keys).',
    )
    parser.add_argument(
        '--remove',
        action='store_true',
        help='Remove unused keys from the locale file (default: disabled).',
    )
    parser.add_argument(
        '--remove-maybe',
        action='store_true',
        help='When used with --remove, also remove maybe-unused keys.',
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='When used with --remove, do not create a .bak backup.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Write unused keys (one per line) to a file instead of stdout.',
    )
    args = parser.parse_args()

    locale_path: Path = args.locale_file
    tsx_root: Path = args.tsx_root
    extensions = tuple(args.ext)

    if not locale_path.exists():
        raise SystemExit(f'Locale file not found: {locale_path}')
    if not tsx_root.exists():
        raise SystemExit(f'Scan root not found: {tsx_root}')

    locale_obj = json.loads(locale_path.read_text(encoding='utf-8'))
    if not isinstance(locale_obj, dict):
        raise SystemExit(f'Expected JSON object at root of {locale_path}')

    locale_keys = set(_flatten_locale_keys(locale_obj))
    result = scan_used_keys(tsx_root=tsx_root, locale_keys=locale_keys, extensions=extensions)

    ignore_res = [re.compile(p) for p in args.ignore_key_regex]
    ignored = {k for k in result.locale_keys if any(r.search(k) for r in ignore_res)}
    used_keys = result.used_keys | ignored
    maybe_used = result.maybe_used - used_keys
    raw_unused = sorted(result.locale_keys - used_keys - maybe_used)

    out_text = '\n'.join(raw_unused) + ('\n' if raw_unused else '')
    if args.output:
        args.output.write_text(out_text, encoding='utf-8')
    else:
        print(out_text, end='')

    maybe_text = '\n'.join(maybe_used) + ('\n' if maybe_used else '')
    if args.maybe_output:
        args.maybe_output.write_text(maybe_text, encoding='utf-8')
    if args.show_maybe and not args.output:
        if maybe_used:
            print('\n# maybe-unused')
            print(maybe_text, end='')

    if args.remove:
        removed = 0
        keys_to_remove = list(raw_unused) + (list(maybe_used) if args.remove_maybe else [])
        for k in keys_to_remove:
            if _delete_key_path(locale_obj, k):
                removed += 1

        if not args.no_backup:
            backup_path = locale_path.with_suffix(locale_path.suffix + '.bak')
            backup_path.write_text(locale_path.read_text(encoding='utf-8'), encoding='utf-8')

        locale_path.write_text(json.dumps(locale_obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

        print(
            f'Removed {removed} key(s), kept {len(result.locale_keys) - removed} (backup: {"none" if args.no_backup else str(backup_path)})'
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
