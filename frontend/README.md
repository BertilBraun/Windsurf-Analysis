# Frontend (React + TypeScript + Vite)

React-based web application integrated with Firebase for authentication, database, and storage.

### Core Technologies
*   **Framework**: React with TypeScript and Vite.
*   **State & Routing**: React hooks and `react-router-dom`.
*   **Backend Integration**: Firebase (Auth, Firestore, Storage) and a custom Cloud Run backend.
*   **Internationalization**: `i18next` supporting English, German, Spanish, and Italian.
*   **Media Processing**: Client-side video transcoding and frame manipulation via Mediabunny.

### Configuration
*   **Environment Variables**: Uses `VITE_` prefixed variables (e.g., Firebase config, `VITE_BACKEND_URL`). Validated at runtime via `src/env.ts`.
*   **Polyfills**: Configured via `vite.config.ts` to support Node.js globals (`Buffer`, `process`, `global`) in the browser.
*   **Port**: Defaults to `5173`.

### Development Commands
*   **Install**: `npm install`
*   **Local Dev**: `npm run dev`
*   **Build**: `npm run build` (Outputs to `dist/`)
*   **Preview Build**: `npm run preview`
*   **Firebase Emulator**: `firebase emulators:start --only hosting`

### Folder Structure
*   **src/**: Core source code including UI components, Firebase initialization, media utilities (WebCodecs player, transcoding), and i18n configuration.
*   **scripts/**: Maintenance utilities, such as `i18n_unused_keys.py` for scanning and cleaning up translation files.

### Deployment
*   **Target**: Firebase Hosting.
*   **Command**: `firebase deploy --only hosting` (Run after `npm run build`).

### TODO
*   Add automated tests for environment variable validation logic.
*   Consolidate custom type definitions into a dedicated `types/` directory.
