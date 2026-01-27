# Backend Service

FastAPI application providing the core API for video analysis, user management, and job tracking. It integrates with Firebase for authentication/database and Modal for heavy processing.

### Core Components

*   main.py: Application entry point, CORS configuration, and router registration.
*   config.py: Environment variable management and global settings (CORS, Firestore, Modal secrets).
*   models.py: Pydantic schemas for Firestore documents and API payloads, including JobRecord, UserRecord, and TrackResult.

### Architecture

*   auth: Handles Firebase ID token verification and internal service-to-service authentication via shared secrets.
*   db: Initializes the Firestore client and provides centralized access to collections (jobs, users, reports).
*   repos: Data access layer managing persistence logic for jobs, users, and feedback reports.
*   routes: API endpoint definitions organized by resource (Jobs, Users, Feedback, Internal).
*   storage: Utilities for serializing and retrieving JSON data from Google Cloud Storage.

### Key Features

*   Job Lifecycle Management: Tracks video processing from upload through stabilization, detection, and tracking.
*   User Quotas: Enforces limits on the number of jobs processed per user.
*   Result Persistence: Stores large tracking results in GCS to bypass Firestore document size limits.
*   Internal Webhooks: Secure endpoints for external workers (Modal) to update job status and submit results.

### Local Development

*   Install dependencies: pip install -r requirements.txt
*   Set up environment variables in a .env file (refer to config.py for keys).
*   Authenticate with GCP: gcloud auth application-default login
*   Run the server: python main.py (defaults to port 8080).

### Deployment

The service is designed for Google Cloud Run (region: europe-west3). It requires Firebase ID tokens for public endpoints and X-Modal-Secret for internal worker communication.

### TODO

*   Implement recursive deletion of job records and GCS results when no users are associated with a job.
*   Re-enable strict email verification for job creation.
*   Add abuse protections for job submission.
*   Add support for non-JSON file formats in storage utilities.
*   Implement repository patterns to further decouple business logic from Firestore-specific syntax.
