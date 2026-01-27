# Authentication

Handles user identity verification and service-to-service security using FastAPI dependencies.

### Firebase Authentication
*   **User Identity**: `User` dataclass containing UID, email, name, and profile picture.
*   **get_current_user**: Dependency that validates a Firebase Bearer token and enforces email verification.
*   **get_current_user_without_email_verification**: Dependency that validates a Firebase Bearer token but allows users with unverified emails.
*   **Initialization**: Automatically initializes the Firebase Admin SDK on module load.

### Internal Authentication
*   **require_modal_secret**: Dependency for securing internal endpoints (e.g., Modal functions).
*   **Shared Secret**: Compares a custom request header against a configured `MODAL_SHARED_SECRET`.
*   **Fail-Closed**: Raises a 500 error if the server-side secret is not configured.

### Configuration Requirements
*   Firebase Admin SDK credentials must be available in the environment.
*   `MODAL_SHARED_SECRET` and the corresponding header alias must be defined in application settings.
