This folder contains the repository layer responsible for data persistence and retrieval using Google Cloud Firestore and Google Cloud Storage (GCS).

### Repositories

*   **JobsRepo**: Manages job metadata and processing results.
    *   Stores job records in Firestore using file checksums as unique IDs.
    *   Handles deterministic GCS object naming for job results (results/{job_id}.json).
    *   Provides methods for creating, updating, and retrieving job status and results.
*   **ReportsRepo**: Handles user-submitted feedback and issue reports.
    *   Supports categories: missed detections, false associations, visual problems, feedback, and others.
    *   Stores reports with user and job associations in Firestore.
*   **UserJobsRepo**: Manages the many-to-many relationship between users and jobs.
    *   Supports soft-deletion of user-job associations.
    *   Provides batch operations for deleting multiple associations.
    *   Tracks which users have access to specific job results.
*   **UserRepo**: Manages user profiles and usage state.
    *   Tracks user consent (terms, privacy, marketing).
    *   Maintains job quotas and increments processed job counts atomically.
    *   Updates activity timestamps and manages basic CRUD for user documents.

### TODO

*   In UserJobsRepo.delete_all_for_user: Implement recursive deletion of job records and GCS results if no other users are associated with a job.
