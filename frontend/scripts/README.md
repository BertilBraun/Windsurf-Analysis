# Frontend Scripts

Utility scripts for frontend maintenance and development.

### i18n_unused_keys.py

Scans source code to identify and optionally remove unused translation keys from locale JSON files.

*   **Usage**: `python3 i18n_unused_keys.py [options]`
*   **Scanning**: Searches for string literals in `.tsx` (default) and `.ts` files that match keys in the locale file.
*   **Key Detection**:
    *   **Unused**: Keys not found in the source code.
    *   **Maybe-unused**: Keys where the parent path is found (e.g., in a template literal or dynamic access) but the full key is not explicitly present.
*   **Features**:
    *   `--remove`: Deletes unused keys from the JSON file.
    *   `--remove-maybe`: Also removes keys flagged as "maybe-unused".
    *   `--ignore-key-regex`: Skips keys matching specific patterns (treats them as used).
    *   `--output` / `--maybe-output`: Saves results to specified text files.
    *   `--no-backup`: Disables the automatic `.bak` file creation when using `--remove`.
*   **Default Paths**:
    *   Locale: `frontend/src/i18n/locales/en.json`
    *   Source Root: `frontend/src`
