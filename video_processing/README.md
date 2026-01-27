# Video Processing Pipeline

Serverless pipeline for windsurfing video analysis, built on Modal and Google Cloud Storage.

### Core Components

*   **Trigger (`main_trigger.py`)**: FastAPI ASGI app providing an internal endpoint to initiate processing jobs.
*   **Orientation (`main_orientation.py`)**: CPU-based service that detects dominant video orientation and applies rotation fixes.
*   **Inference (`main_inference.py`)**: GPU-accelerated (T4) YOLO object detection for identifying riders and equipment.
*   **Tracking (`main_tracking.py`)**: Orchestrates the final analysis stages:
    *   **Stabilization**: Computes camera motion transforms using masked video stabilization.
    *   **Appearance**: Extracts Re-ID embeddings for detected objects.
    *   **ILP Tracking**: Links detections into persistent tracks using Integer Linear Programming.
*   **GCS IO (`gcs_io.py`)**: Utilities for managing video assets in Google Cloud Storage, including authenticated downloads, uploads, and cleanup.

### Infrastructure & Deployment

*   **Deployment (`deploy.py`)**: Unified script to deploy the trigger, inference, orientation, and tracking apps under a single Modal namespace.
*   **Configuration (`config.py`)**: Manages environment variables for Cloud Run callbacks and shared secrets.
*   **Webhooks**: Reports job progress (orientation, detection, tracking, etc.) and final results back to a central backend via HTTP POST.

### Subdirectories

*   **[api/](./api)**: Internal business logic and clients for external service integrations.
*   **[inference/](./inference)**: Core execution engine, including tracking algorithms and hyperparameter optimization tools.
*   **[scripts/](./scripts)**: Local utilities for pipeline testing, visualization rendering, and stabilization benchmarking.

### TODO

*   Standardize metadata formats between optimization tools and tracking outputs.
*   Implement API controllers and data models for video metadata in the `api/` layer.
*   Standardize CLI arguments between local scripts and the cloud pipeline.
*   Add batch processing support for local pipeline scripts.
