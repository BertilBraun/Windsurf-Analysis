This folder contains the core logic for object detection and multi-object tracking (MOT), specifically optimized for surfers and windsurfers. It includes multiple tracking algorithms ranging from greedy heuristics to global optimization via Integer Linear Programming (ILP).

### Core Components

*   **SurferDetector**: A two-pass detection pipeline.
    *   Pass 1: Runs YOLO (pose-enabled) to extract bounding boxes and keypoints (boom, mast tip).
    *   Pass 2: Extracts ReID embeddings for detected crops using various models (OSNet, ViT, or Color Histograms).
*   **ILP Tracking Suite**: Global optimization trackers that link track fragments by solving an assignment problem.
    *   **ILPGraphSolver**: A PuLP-based solver that minimizes costs for fragment links, starts, ends, and discards.
    *   **ILPTracker**: Uses a weighted cost function combining motion (Kalman Filter Mahalanobis distance), appearance (ReID similarity), and temporal gaps. Includes spatial costs to favor tracks starting/ending at image borders.
    *   **IterativeILPTracker**: Runs multiple optimization passes with a scheduled start-cost to refine track continuity and handle internal splits based on motion/appearance anomalies.
    *   **DiscreteOptTracker**: A simplified ILP implementation focusing on IoU and embedding cohesion within temporal windows.
*   **OCSortEmbedTracker**: An implementation of OC-SORT (Observation-Centric SORT) enhanced with ReID embeddings, BYTE-style low-confidence association, and Camera Motion Compensation (CMC).
*   **GreedyTracker**: A simple association engine that merges tracklets based on highest average-embedding cosine similarity and IoU.

### Post-Processing & Smoothing

*   **TrackRTSSmoothing**: Applies Rauch-Tung-Striebel (RTS) smoothing. It uses a forward Kalman Filter pass with CMC followed by a backward smoothing pass to fill gaps and stabilize trajectories.
*   **TrackFiltering**: Removes spurious tracks based on minimum frame count and detection density (e.g., requiring at least 30% detection hits over the track duration).
*   **RenderableTracks**: Prepares tracks for stable video rendering.
    *   **Anchors**: Calculates per-frame anchor points biased toward the mast/boom segment to keep the subject centered.
    *   **Scales**: Computes normalized crop heights to ensure the mast occupies a consistent percentage of the frame.

### Utilities

*   **CMC (Camera Motion Compensation)**: Integrated into trackers and smoothers to account for camera pan, tilt, and zoom using frame-to-frame transforms.
*   **KalmanBox**: A SORT-style Kalman Filter for bounding box state estimation (position, velocity, and aspect ratio).
*   **Tracker Protocol**: Defines the standard interface for all tracking implementations.

### Subfolders

*   **preprocessing**: Utilities for initial track cleaning and greedy stitching.
*   **reid**: Implementations for visual feature extraction, including deep learning (OSNet, ViT) and color-based histograms.

### TODO

*   Standardize the `Tracker` protocol usage across all implementations.
*   Refine the spatial start/end cost interpolation logic in `ILPTracker`.
*   Validate the performance impact of `dedup_enable` in `OCSortEmbed`.
*   Address the "TODO prefer to merge longer tracks?" note in `GreedyTracker`.
