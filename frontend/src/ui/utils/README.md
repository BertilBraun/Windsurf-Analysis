Utility functions for browser API interactions, file processing, and cross-tab synchronization.

### Analytics & Tracking
*   **analytics.ts**: Google Analytics (gtag) integration with consent management, page view tracking, and automatic click monitoring for interactive elements.
*   **assert.ts**: Runtime validation utility for condition checking.

### Browser Storage & Filesystem
*   **fsAccess.ts**: Helpers for the File System Access API, including recursive file listing, permission prompts, and relative path resolution.
*   **idb.ts**: IndexedDB wrapper for persisting application state, directory handles, file snapshots, and thumbnail blobs.
*   **localFileIndex.ts**: Utilities for file fingerprinting using a sampled SHA-256 strategy to efficiently hash large files by skipping blocks.

### Cross-Tab Synchronization
*   **crossTabChannel.ts**: Low-level communication layer using `BroadcastChannel` with `localStorage` fallback for multi-tab synchronization.
*   **ingressDirectorySync.ts**: Signals changes to the ingress directory across tabs.
*   **ingressScannerSync.ts**: Shares scanner state (active status, progress, errors) and broadcasts commands (e.g., retry failed) between tabs.
*   **localFileSnapshotSync.ts**: Signals updates to the local file index across tabs.

### Async & Data Processing
*   **concurrency.ts**: Hardware-aware concurrency management and `mapLimit` for processing async tasks with a maximum parallel execution cap.
*   **uploader.ts**: Manages video uploads to Firebase Storage, including job creation, quota checks, parallel upload limiting, and progress reporting.
*   **clamp.ts**: Numeric utility to restrict values within a specific range.

### TODO
*   Re-enable video preprocessing in `uploader.ts`.
