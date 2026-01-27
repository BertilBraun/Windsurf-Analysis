API endpoint definitions for the backend, organized by resource.

### Feedback
- `POST /feedback`: Submits user feedback messages to the reports repository.

### Internal Jobs
- Restricted endpoints for internal workers (e.g., Modal) using secret-based authentication.
- `POST /internal/jobs/{job_id}/status`: Updates job execution status and error messages.
- `POST /internal/jobs/{job_id}/results`: Submits final processing results, including object tracks, orientation, and stabilization transforms. Increments user processing counts upon success.

### Jobs
- Manages the lifecycle of video processing tasks.
- `POST /jobs`: Creates a job record. Uses checksum deduplication to reuse existing results and enforces user job quotas.
- `POST /jobs/{job_id}/upload/complete`: Finalizes the upload process and triggers external processing workers via Modal.
- `GET /jobs/{job_id}`: Retrieves job status and detailed results (tracks and transforms).
- `DELETE /jobs/{job_id}` & `POST /jobs/bulk-delete`: Marks user-job associations as deleted.
- `POST /jobs/{job_id}/report`: Files a report for a specific job issue.

### Users
- Manages user profiles and account state.
- `POST /users/{user_id}`: Creates a new user record and tracks consent for terms of service.
- `PATCH /users/{user_id}/consent`: Updates marketing and terms of service consent.
- `GET /users/{user_id}`: Retrieves user metadata and processing statistics.
- `DELETE /users/{user_id}`: Performs full account deletion, including Firebase Auth removal and associated user-job links.

### TODO
- Implement recursive deletion of job result documents when no users reference a job.
- Re-enable strict email verification for job creation (currently relaxed for onboarding).
- Add abuse protections for job submission.
