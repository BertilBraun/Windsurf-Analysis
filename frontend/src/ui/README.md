This folder contains the core UI layer of the application, including the root entry point, global state providers, and specialized subdirectories for components and logic.

### Core Files

*   **App.tsx**: The root React component.
    *   Initializes global providers (`AuthProvider`) and the application `Router`.
    *   Handles global side effects: analytics initialization, click tracking, and synchronizing HTML language attributes.
    *   Renders global banners for browser support and analytics consent.
*   **types.ts**: Centralized TypeScript definitions for the UI.
    *   **Job Models**: `JobStatus` (uploading to succeeded/failed), `JobSummary`, and `JobDetail`.
    *   **Tracking Data**: `Track` and `TrackDetection` for object movement and bounding boxes.
    *   **Video Metadata**: `StabilizationTransform` (dx, dy, da) and `UploadQuality` presets.

### Subfolder Architecture

*   **auth/**: Firebase authentication logic, context providers, and authorized fetch wrappers.
*   **components/**: Reusable UI elements (Button, Modal), layout shells, and feature-specific widgets (JobList, IngressWidget).
*   **hooks/**: Custom hooks for data synchronization (Firestore/IndexedDB), local file indexing, and UI interactions.
*   **player/**: High-performance video player using WebCodecs, including canvas rendering, stabilization, and MP4 export.
*   **routes/**: Routing configuration, including protected route logic and session management for analyzer and demo modes.
*   **screens/**: Top-level page components (Home, Analyzer, Demo, Legal, etc.).
*   **utils/**: Low-level utilities for browser APIs (File System Access, IndexedDB), cross-tab communication, and file uploading.
