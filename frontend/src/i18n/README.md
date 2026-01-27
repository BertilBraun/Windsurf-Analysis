# Internationalization (i18n)

This folder manages application-wide translations and language settings using `i18next` and `react-i18next`.

### Features
*   **Supported Languages**: English (`en`), German (`de`), Spanish (`es`), and Italian (`it`).
*   **Persistence**: Automatically saves and retrieves the user's language preference from `localStorage` using the key `gybelock.language`.
*   **Language Detection**:
    1. Checks `localStorage` for a saved preference.
    2. Falls back to the browser's navigator language.
    3. Defaults to English (`en`) if no match is found.
*   **Normalization**: Automatically handles complex language codes (e.g., converting `en-US` or `en_GB` to `en`).
*   **React Integration**: Configured with `initReactI18next` and `useSuspense: false` for immediate rendering.

### Structure
*   `index.ts`: Main configuration, initialization logic, and language normalization.
*   `locales/`: Contains JSON translation files for each supported language.

### TODO
*   Add automated tests for language normalization and fallback logic.
*   Implement a mechanism for lazy-loading translation files if the bundle size grows significantly.
