# Windsurf Analysis / GybeLock — Documentation

This repo contains two closely related things:

1. **The windsurf video processing pipeline** (Python): orientation fixing → YOLO pose detection → camera-motion stabilization → appearance features → tracking → smoothed “renderable” tracks.
2. **The GybeLock web app (production MVP)**: a React frontend that lets users select a local “ingress folder”, uploads videos for processing, and plays results in an interactive player.

If you’re trying to understand “what the application does” end-to-end, start here.

---

## What the application does (end-to-end)

### Purpose

GybeLock/Windsurf Analysis is built for **beach-shot / shore-shot windsurf footage** (tele/zoom camera from land). It helps you:

- upload session footage for analysis,
- automatically detect + track riders over time,
- review results in a player that supports an **overview mode** and a **focused mode** (click a rider → get a stable zoomed view),
- report failures so the pipeline can be improved.

It is *not* designed for GoPro/action-cam POV footage.

### Processing flow (production MVP)

1. **Frontend (browser)**: user selects an “ingress folder” (File System Access API) and drops `.mp4` files into it.
2. **Upload**: the frontend creates a job via the backend and uploads the video to **Firebase Storage** (path like `uploads/<uid>/<job_id>.mp4`).
3. **Backend (FastAPI)**: validates ownership/quota, stores job metadata in **Firestore**, then triggers **Modal**.
4. **Modal pipeline**:
   - orientation normalization,
   - YOLO pose detection (bbox + keypoints),
   - stabilization transform estimation (masked optical flow),
   - appearance feature extraction,
   - tracking + smoothing + “renderable track” preparation,
   - posts results back to the backend via internal endpoints.
5. **Results**: backend stores the results JSON in **Firebase Storage / GCS** (to avoid Firestore document size limits) and updates job state.
6. **Player**: frontend polls Firestore for job state; when a job succeeds it fetches results (tracks + stabilization transforms) and renders the interactive player.

### What is implemented vs. planned

Implemented today (in this repo):

- React + Firebase frontend (`frontend/`)
- FastAPI backend using Firebase Auth + Firestore + Firebase Storage/GCS (`backend/`)
- Modal deployment for inference/tracking (`video_processing/`)
- Local reproduction runner for debugging/training iteration (`video_processing/scripts/…`)
- Training/annotation tooling (`train/…`)

Planned / historical (kept for reference, but not fully aligned with current code):

- Earlier “production requirements” that assumed **Neon Postgres + R2 + Basic Auth** (`documentation/PRODUCTION_REQUIREMENTS.md`)
- Early player and UI specs that predate the current web player implementation (`documentation/PLAYER_REQUIREMENTS.md`, `documentation/GybeLock-UI-Plan.md`)

Those documents are still useful as design context, but they should not be treated as source-of-truth for the current deployed MVP.

---

## Documentation map (all first‑party Markdown)

Note: the repo contains many third‑party `README.md` files under `node_modules/`; those are intentionally excluded here.

### Top-level

- `README.md` — entrypoint: what this repo is, quick starts, links.
- `FEEDBACK_RUNBOOK.md` — “when a user video doesn’t work”: reproduce → triage → improve → re-test → deploy.

### Web app (GybeLock)

- `frontend/README.md` — run the React app locally and deploy to Firebase Hosting.
- `backend/README.md` — run/deploy the FastAPI backend and required env vars.
- `documentation/FIREBASE_SETUP.md` — Firebase project setup + local dev wiring.
- `documentation/DEPLOYMENT.md` — end-to-end deployment checklist (Firebase Hosting + Cloud Run + Modal).
- `documentation/ANALYZER_TUTORIAL.md` — end-user walkthrough: ingress folder → upload → open player → export/report.
- `documentation/ANALYZER_FAQ.md` — end-user FAQ: supported footage, “VIDEO FILE NOT FOUND”, quotas, etc.
- `documentation/GybeLock-UI-Plan.md` — UI/brand plan (design doc; may reference refactors not yet done).

### Video processing pipeline (Python)

- `frontend/public/TECHNICAL.md` — technical deep dive (developer-facing; used by the web app as a static doc page).
- `documentation/TECHNICAL.md` — short technical entrypoint + pointers to deeper docs.
- `video_processing/inference/documentation.md` — tracking pipeline doc (code-oriented, focuses on the tracking stages).
- `video_processing/inference/src/motion/README.md` — notes + images for motion compensation / Kalman tracking debugging.

### Training / datasets

- `train/detection/README.md` — bbox + keypoint annotation and YOLO training workflow.
- `train/rotation-classification/README.md` — orientation classifier training (0/90/180/270 degrees).

---

## Where to go next

- Want to run the web app locally? Start with `documentation/FIREBASE_SETUP.md`.
- Want to understand the pipeline? Start with `frontend/public/TECHNICAL.md`.
- Got a failing video and want to debug/iterate? Start with `FEEDBACK_RUNBOOK.md`.

