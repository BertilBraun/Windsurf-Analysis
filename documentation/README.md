# Documentation

Central repository for project guides, technical specifications, and operational runbooks for the Windsurf Analysis / GybeLock platform.

### Setup & Deployment
*   **FIREBASE_SETUP.md**: Instructions for configuring Firebase project, Firestore, and local development environment.
*   **DEPLOYMENT.md**: End-to-end checklist for deploying the React frontend (Firebase Hosting), FastAPI backend (Cloud Run), and processing pipeline (Modal).

### User Guides
*   **ANALYZER_TUTORIAL.md**: Walkthrough for the ingress workflow, including folder selection, uploading, and using the interactive player.
*   **ANALYZER_FAQ.md**: Troubleshooting and common questions regarding supported footage, file errors, and quotas.

### Technical Specifications
*   **TECHNICAL.md**: High-level entry point for the video processing pipeline architecture.
*   **GybeLock-UI-Plan.md**: Design document covering UI components, branding, and planned interface refactors.
*   **PRODUCTION_REQUIREMENTS.md**: Historical requirements document (Note: some details regarding Postgres/R2 are superseded by the current Firebase MVP).
*   **PLAYER_REQUIREMENTS.md**: Early specifications for the video player and overlay system.

### Subfolders
*   **scripts/**: Python utilities for local video processing:
    *   **pose_video.py**: YOLOv8-pose detection and keypoint extraction.
    *   **ws_ingress.py**: Interactive `mpv`-based footage review and cutting.
    *   **ws_reduce_size.py**: Batch video compression (archival or messaging profiles).
    *   **ws_stabilize.py**: Parallel video stabilization using FFmpeg.

### TODO
*   Consolidate overlapping technical documentation between `documentation/` and `frontend/public/`.
*   Update legacy requirements files to align with the current Firebase/Modal architecture.
*   Add `requirements.txt` for the Python scripts.
