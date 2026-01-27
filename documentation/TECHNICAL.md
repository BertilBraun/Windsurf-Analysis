# Technical entrypoint

This file is intentionally short and points to the docs that are kept most up-to-date.

## Start with

- `frontend/public/TECHNICAL.md` — end-to-end technical deep dive (pipeline + web app wiring).

## Pipeline docs

- `video_processing/inference/documentation.md` — tracking pipeline stages and rationale (code-oriented).
- `FEEDBACK_RUNBOOK.md` — how to debug a failing sample and iterate on models/tracking/params.

## Training docs

- `train/detection/README.md` — detection/pose dataset workflow + training scripts.
- `train/rotation-classification/README.md` — orientation model training.

## Note on older documents

Some older writeups still exist in this repo as historical context, but they may describe earlier approaches
(different models/features, different production architecture). `documentation/README.md` calls out which docs
are “current” vs “historical/spec”.
