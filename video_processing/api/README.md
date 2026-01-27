# API Layer

Core interface and business logic for the video processing system, designed to separate external integrations from internal workflows.

### Subdirectories
* **clients/**: Implementations for communicating with external services (storage, transcoding, metadata).
* **services/**: Business logic layer for orchestrating workflows like transcoding, thumbnail generation, and file persistence.

### Key Responsibilities
* Managing video processing workflows and status tracking.
* Interfacing with storage providers and message queues.
* Decoupling API endpoints from underlying data access and processing implementations.

### TODO
* Implement API controllers and routes to expose service functionality.
* Define data models for video metadata and processing status.
* Integrate authentication and authorization middleware.
* Implement `VideoService` and `StorageService` as defined in the services layer.
* Implement storage provider clients (S3, Azure, etc.) in the clients layer.
