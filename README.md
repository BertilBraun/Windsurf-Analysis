# Windsurf Analysis / GybeLock

This repo turns **beach-shot / shore-shot windsurf footage** into:

- rider detections + tracks,
- stabilization transforms (for a smoother overview),
- an interactive player that lets you click a rider to get a stable “focused view”.

It contains both:

- a **production MVP web app** (“GybeLock”) built with React + Firebase + FastAPI + Modal, and
- the underlying **Python video processing pipeline** plus training/optimization tooling.

![Demo](documentation/processed.gif)

## Start here

- `documentation/README.md` — what the app does, what’s implemented, and where each doc lives.

## Quick starts

### 1) Run the web app locally (GybeLock)

- Follow `documentation/FIREBASE_SETUP.md` to configure Firebase.
- Then run:

```bash
# Backend (FastAPI)
pip install -r backend/requirements.txt
python backend/main.py

# Frontend (React + Vite)
cd frontend
npm install
npm run dev
```

### 2) Reproduce the pipeline locally (debugging / iteration)

The most faithful local reproduction of the production pipeline is:

```bash
python video_processing/scripts/local_modal_pipeline_player.py --input-video tmp/test/input.mp4 --output-dir tmp/test/out
```

See `FEEDBACK_RUNBOOK.md` for the workflow around failing samples.

### 3) Deploy (Firebase Hosting + Cloud Run + Modal)

Use `documentation/DEPLOYMENT.md`.

## License

No license file is currently included in this repo. If you plan to redistribute or commercialize this code, clarify licensing first.
