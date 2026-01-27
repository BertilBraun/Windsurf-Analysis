Core source code for the frontend application, including the React entry point, service initializations, and core utilities.

### Core Files

*   **main.tsx**: The application entry point. Initializes the React root, global CSS, and wraps the app in `I18nextProvider` and `BrowserRouter`.
*   **firebase.ts**: Initializes Firebase services (Authentication, Firestore, and Storage) and exports the `backendUrl` configuration.
*   **env.ts**: Utility for retrieving and validating required environment variables from `import.meta.env`. Throws errors for missing or placeholder values.
*   **file-system-access.d.ts**: Type definitions for the File System Access API, enabling directory picking and permission management.
*   **sha.js.d.ts**: Type definitions for the `sha.js` library used for SHA-256 hashing.
*   **vite-env.d.ts**: Standard Vite client type references.

### Subfolder Overview

*   **ui/**: The main React layer, containing the root `App` component, routing, state management, hooks, and a high-performance WebCodecs video player.
*   **preprocess/**: Utilities for client-side video transcoding, resizing, and frame-by-frame manipulation using the Mediabunny library.
*   **media/**: Tools for extracting video packet metadata (timestamps, keyframes) and performing binary searches on packet indices.
*   **i18n/**: Internationalization configuration supporting English, German, Spanish, and Italian with automatic language detection and persistence.

### Environment Requirements

The application requires several `VITE_` prefixed environment variables for Firebase and backend connectivity. These are validated at runtime via `env.ts`. Required variables include:
*   Firebase configuration (API Key, Auth Domain, Project ID, etc.)
*   `VITE_BACKEND_URL`
*   `VITE_FIREBASE_DATABASE_ID`

### TODO

*   Add automated tests for environment variable validation logic.
*   Consolidate custom type definitions into a dedicated `types/` directory if the number of `.d.ts` files increases.
