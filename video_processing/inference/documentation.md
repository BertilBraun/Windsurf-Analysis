# Windsurf Tracking Pipeline Documentation

This document outlines the architecture and algorithms used in the windsurf tracking system located in `video_processing/inference/src`.

## Overview

The tracking pipeline is orchestrated by `WindsurfingVideoProcessor` in `windsurf_video_processor.py`. It transforms raw video input into stable, identified tracks of surfers using a multi-stage approach:

1.  **Object Detection**: an Ultralytics YOLO **pose** model produces bboxes + keypoints per frame.
2.  **Camera Motion Compensation (CMC)**: Global Motion Compensation (GMC) calculates camera movement to allow tracking in a stabilized coordinate system.
3.  **Tracking Pipeline**:
    *   **Preprocessing**: Greedy stitching of obvious connections.
    *   **Global Optimization**: Iterative Integer Linear Programming (ILP) solver to resolve complex associations.
    *   **Post-Processing**: RTS (Rauch-Tung-Striebel) Smoothing and filtering.

---

## 1. Orchestration (`windsurf_video_processor.py`)

The `WindsurfingVideoProcessor` class manages the entire lifecycle of a video processing job.

### Key Responsibilities
*   **Setup**: Initializes the detector, executor pools, and output directories.
*   **Execution**: `process_video()` runs the pipeline sequentially:
    1.  **Detection**: Runs `SurferDetector` to get bounding boxes for all frames.
    2.  **Stabilization**: Computes affine transforms (GMC) to map frame $t$ to $t-1$.
    3.  **Tracking**: Calls `_process_detections_into_tracks` which passes data through a chain of trackers.
    4.  **Export**: Saves metadata (`.tracks.pkl`) and optionally renders debug/annotated videos.

### The Tracking Chain
The tracking logic is modularized as a sequence of "Trackers" that refine the list of tracks:
```python
trackers=[
    TrackPreProcessor(...),       # Step 1: Greedy Stitching
    IterativeILPTracker(...),     # Step 2: Global Optimization
    TrackPostProcessing(...),     # Step 3: Smoothing & Filtering
]
```

---

## 2. Preprocessing (`tracking/preprocessing/preprocessor.py`)

**Goal**: Reduce the problem size for the expensive ILP solver by linking "obvious" connections first.

The `TrackPreProcessor` uses a `GreedyTrackStitcher` to link detections into "tracklets" (fragments).

### Approach
*   **Greedy Matching**: Iterates through detections and links them if they satisfy strict spatiotemporal and appearance constraints.
*   **Thresholds**: Uses "Strict" and "Loose" probability thresholds for:
    *   **Appearance**: Visual similarity (likely using embeddings/histograms).
    *   **Motion**: Spatial proximity.
*   **Result**: Instead of thousands of individual detections, the next stage receives a smaller number of high-confidence track fragments.

---

## 3. Global Optimization (`tracking/iterative_ilp_tracker.py`)

**Goal**: Link track fragments into complete trajectories, handling occlusions, gaps, and crossings.

The `IterativeILPTracker` is the core of the system. It builds a graph where nodes are track fragments and edges are possible links, then solves for the optimal path.

### Algorithm: Iterative Solve & Split
Unlike a standard tracker that runs once, this runs in a loop (`max_optimization_iterations`):

1.  **Graph Construction**:
    *   **Nodes**: Existing track fragments.
    *   **Edges**: Created between fragment $A$ (end) and fragment $B$ (start) if they are within `MAX_OVERLAP_LENGTH_SECONDS`.
    *   **Costs**: Calculated as a weighted sum of Negative Log-Likelihoods (NLL):
        *   **Motion Cost**: Uses a Kalman Filter (KF) to predict $A$'s position at $B$'s start time.
            *   Crucially, it applies **Camera Motion Compensation (CMC)** to transform coordinates into a common reference frame during prediction.
            *   Computes Mahalanobis distance between prediction and observation.
        *   **Appearance Cost**: Compares visual embeddings (LAB color histograms) using Chi-squared distance.
        *   **Gap Cost**: Penalizes missing frames (time gaps).

2.  **ILP Optimization**:
    *   Solves the global assignment problem (likely using min-cost flow or linear assignment) to minimize the total cost.
    *   **Start Cost Schedule**: The cost to start a new track (`w_start`) increases with each iteration. This initially allows fragmented tracks (low penalty) and gradually forces the solver to merge them if possible.

3.  **Track Splitting (`_maybe_split_tracks`)**:
    *   After merging, the system sanity-checks the new tracks.
    *   It iterates through the detections inside a track and calculates "internal" motion and appearance consistency.
    *   If a sudden jump in position (high motion NLL) or change in color (high appearance NLL) is detected *within* a track, it breaks the track apart.
    *   **Why?** This corrects bad merges made by the greedy preprocessor or previous ILP iterations.

### Key Concepts
*   **Iterative Refinement**: The loop (Build -> Solve -> Split) allows the system to recover from errors. A bad merge creates a high-cost internal link, which the Split step breaks, allowing the next Build/Solve step to find a better alternative.
*   **Robustness**: By combining Motion (geometry) and Appearance (visuals), it handles cases where surfers cross paths (visuals help) or look similar (geometry helps).

---

## 4. Post-Processing (`tracking/track_processing.py`)

**Goal**: Clean up the tracks and produce smooth, continuous trajectories.

The `TrackPostProcessing` class runs a pipeline of filters:

### 1. Filtering (`TrackFiltering`)
*   Removes tracks that are too short or have too few detections relative to their duration.
*   Criteria: `MIN_FRAME_PERCENTAGE` (e.g., track must have detections for at least X% of its duration).

### 2. RTS Smoothing (`TrackRTSSmoothing`)
This replaces simple linear interpolation with a theoretically optimal **Rauch-Tung-Striebel (RTS) Smoother**.

*   **Forward Pass**: Runs a standard Kalman Filter forward through time.
    *   Predicts state $x_{t|t-1}$.
    *   Updates with measurement $x_{t|t}$.
    *   Applies CMC (transform from $t-1$ to $t$) at every step.
*   **Backward Pass**: Runs backward from the last frame to the first.
    *   Smooths the estimate $x_{t|N}$ using information from the future.
*   **Gap Filling**:
    *   For frames with no detection, the RTS smoother provides an optimal estimate based on past and future data.
    *   Generates "Synthesized" detections for every frame in the track's life, ensuring the output video has no flickering or missing boxes.

### 3. Relabeling (`TrackRelabeling`)
*   Simply re-indexes tracks from 1 to $N$ for clean output ID assignment.

---

## Summary of Data Flow

1.  **Raw Video** $\rightarrow$ **YOLO** $\rightarrow$ `List[Detection]`
2.  `List[Detection]` $\rightarrow$ **PreProcessor** $\rightarrow$ `List[Track]` (fragments)
3.  `List[Track]` $\rightarrow$ **IterativeILP** $\rightarrow$ `List[Track]` (global consistent tracks)
4.  `List[Track]` $\rightarrow$ **RTS Smoother** $\rightarrow$ `List[Track]` (dense, smoothed trajectories)

