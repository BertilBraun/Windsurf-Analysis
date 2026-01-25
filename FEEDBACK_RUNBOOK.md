# Feedback Runbook: when user videos "don't work"

This is the "future me" workflow when someone sends a broken video sample. The goal is to: (1) reproduce, (2) decide what subsystem is at fault, (3) fix via data/hparams/code, (4) re-test on the same sample, (5) deploy.

Preferred: use the VS Code tasks in `.vscode/tasks.json` for the repeatable steps.

---

## Fast path: VS Code tasks (preferred)

Run via **Command Palette** -> "Tasks: Run Task", then pick:

- Reproduce locally (Modal-like): `local_modal_pipeline` (note: currently hard-coded to a local file path; edit it when needed)
- Create golden tracklet labels: `annotate_tracklets`
- Optimize tracking:
  - `optimize_preprocessor` (greedy tracklet stitching; apply results in `video_processing/inference/src/settings.py`)
  - `optimize_iter_ilp` (optimizes `ILPTracker` params; apply results in `video_processing/inference/src/tracking/ilp_tracker.py`)
  - `optimize_ILP` (optimizes `DiscreteOptTracker`, not the production `ILPTracker`)
  - `optimize_BoTSort` (currently not implemented in `video_processing/inference/optimization/optimize_tracker.py`)
- Compare/evaluate: `evaluate_tracker`, `compare_trackers`, `compare_preprocessor`
- Deploy:
  - `deploy_video_processing` (Modal: `video_processing/deploy.py`)
  - `deploy_backend`, `deploy_frontend`

The tasks prompt for:
- `golden-dir` (default `tmp/golden`)
- `video-dir` (default `tmp/test/*.mp4`)

---

## 1) Reproduce locally (always first)

Closest local reproduction of the production pipeline:
- `video_processing/scripts/local_modal_pipeline_player.py`

Typical:

```bash
python video_processing/scripts/local_modal_pipeline_player.py --input-video tmp/feedback/XXX/input.mp4 --output-dir tmp/feedback/XXX/out
```

Useful flags:
- `--limit-frames 500` (iterate fast)
- `--yolo-model-path <path-to-candidate.pt>` (try new pose weights without changing defaults)
- `--skip-orientation` (if you know it's already upright)
- `--stabilizer gmc|masked_vidstab|none` (diagnose stabilization/crop issues)
- `--no-player` (batch/debug mode)

Triage heuristics:
- Bad bbox / missing detections -> detection/pose model or thresholds.
- Keypoints wrong on a specific distribution (e.g. wave sails) -> add keypoint data + retrain pose.
- IDs swap/merge/split while detections look fine -> tracking (preprocessor/ILP params or tracker logic).

---

## 2) Pose/keypoints failures (example: wave sails)

### 2.1 Add/repair training samples

If you don't already have bbox labels for the failing frames:

```bash
python train/detection/annotator.py tmp/feedback/XXX train/detection/windsurf_dataset --samples 300
python train/detection/annotation_editor.py train/detection/windsurf_dataset
```

### 2.2 Annotate keypoints

```bash
python train/detection/annotator_keypoints_fullframe.py --src train/detection/windsurf_dataset --out train/detection/pose_projects/boom_mast_v1 --show-annotated
```

### 2.3 Train pose (ideally on Vast.ai)

Preferred: `train/detection/quickstart_train.sh` (it encodes the "what I usually run").

On a Vast.ai box (e.g. RTX 3060-ish):

```bash
pip install -r requirements.txt
bash train/detection/quickstart_train.sh
```

Notes:
- `train/detection/quickstart_train.sh` is written like a "fresh machine" script (it starts with `git clone ...` + `git checkout production`). Run it from a clean directory on the Vast box, or edit it to skip the clone step if you're already in the repo.
- `train/detection/quickstart_train.sh` uses `--device 0` and runs a long training for pose (500 epochs), then also runs detection training.
- Artifacts usually land under `train/detection/runs/...` (Ultralytics default).

### 2.4 Validate on the failing sample

```bash
python video_processing/scripts/local_modal_pipeline_player.py --input-video tmp/feedback/XXX/input.mp4 --output-dir tmp/feedback/XXX/out_pose_candidate --yolo-model-path train/detection/runs/<pose_run>/weights/best.pt
```

### 2.5 Deploy new pose weights (Modal)

Fastest is to overwrite:
- `video_processing/inference/weights/yolo_models/windsurfing_pose/best.pt`

Then redeploy Modal:

```bash
python video_processing/deploy.py
```

If you want versioned weights (for rollback), be aware:
- Production selection uses the `yolo_model` string sent by the frontend: `frontend/src/ui/utils/uploader.ts`
- Modal packaging currently whitelists `best.pt` only (see `ignore_files(...)` in `video_processing/main_inference.py`), so versioned `.pt` files won't be included unless you update that filter.

---

## 3) Tracking failures (example: ID switches)

### 3.1 Create (or extend) golden tracklet labels

Preferred: VS Code task `annotate_tracklets`.

CLI form:

```bash
python video_processing/inference/optimization/annotate_tracklets.py --output-dir tmp/golden/id_switches tmp/feedback/XXX/input.mp4
```

Produces:
- `tmp/golden/id_switches/<video_stem>.golden.tracks.pkl`

### 3.2 Optimize greedy preprocessor

Preferred: VS Code task `optimize_preprocessor`.

CLI form:

```bash
python video_processing/inference/optimization/optimize_tracker.py --mode preprocessor --golden-dir tmp/golden/id_switches --trials 200 --workers 0
```

Apply results in:
- `video_processing/inference/src/settings.py` (the `GREEDY_PREPROCESSOR_*` constants)

### 3.3 Optimize ILPTracker parameters (production ILP tracker)

Preferred: VS Code task `optimize_iter_ilp`.

CLI form:

```bash
python video_processing/inference/optimization/optimize_tracker.py --mode iter_ilp --golden-dir tmp/golden/id_switches --trials 200 --workers 0
```

Apply results in:
- `video_processing/inference/src/tracking/ilp_tracker.py` (`ILPTracker.__init__` defaults)

### 3.4 If optimization isn't enough: add hard gates

Primary location:
- `video_processing/inference/src/tracking/ilp_tracker.py` (`ILPTracker._build_fragment_graph(...)`, `_motion_nll(...)`)

Typical "hard gate" ideas:
- cap Mahalanobis distance / velocity vs gap
- min IOU gate (in stabilized coords)
- forbid links that are more expensive than end+start (dominance pruning already exists in `ILPTracker`)

After changes: re-run the same broken sample locally + re-score on goldens.

### 3.5 Deploy tracking changes (Modal)

```bash
python video_processing/deploy.py
```

---

## 4) Iterative ILP tracker: reality check / TODO

`video_processing/inference/src/tracking/iterative_ilp_tracker.py` is currently behind `video_processing/inference/src/tracking/ilp_tracker.py`.

Before trying to "switch production to iterative ILP", bring it up to feature parity first. Concrete gaps to check/port (compare the two files):

- Discarding short tracklets: `ILPTracker._compute_discard_costs(...)` + `allow_discard_short_tracklets`, `discard_*` params
- Dominance pruning during edge construction (present in `ILPTracker._build_fragment_graph(...)`)
- `max_outgoing_links` plumbing into `ILPGraphSolver(...)`
- Default parameter choices (the iterative defaults differ and can change behavior a lot: `use_position_only`, weights, start/end schedule)

Also note: the Optuna mode name `iter_ilp` in `video_processing/inference/optimization/optimize_tracker.py` currently optimizes `ILPTracker`, not `IterativeILPTracker`. If you want to optimize the iterative tracker, add a new mode or change that instantiation.

---

## 5) "Crops feel unstable" debugging (where to look)

"Unstable crops" usually means the per-track anchor/scale is wobbling, or the post-processed bbox is wobbling.

Two key places:

1) RTS smoothing (bbox interpolation + smoothing)
   - Code: `video_processing/inference/src/tracking/track_processing.py`
   - Knobs: `video_processing/inference/src/settings.py`
     - `TRACK_RTS_ENABLE_BACKWARD_SMOOTHER` (if this feels like "rubber banding", try turning it off)
     - `TRACK_RTS_PROC_STD_WEIGHT_POS`, `TRACK_RTS_PROC_STD_WEIGHT_VEL`
     - `TRACK_RTS_MEAS_STD_WEIGHT_POS`, `TRACK_RTS_MEAS_STD_WEIGHT_SIZE`

2) Anchor/scale preparation (used for focused view stability)
   - Code: `video_processing/inference/src/tracking/renderable_tracks.py` (`prepare_renderable_tracks(...)`)
   - Knobs (currently function params/defaults):
     - `missing_grace_frames`
     - `mast_smooth_radius`
     - `target_mast_fill`

Practical workflow:
- Reproduce locally on the exact failing segment (`--limit-frames ...`).
- Change one knob at a time, re-run, and see if the focused view stabilizes without cutting off the mast tip.
