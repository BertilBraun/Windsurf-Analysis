# Database Client

Manages the connection and interface with the Google Cloud Firestore database.

### Core Components

*   **Firestore Client**: Initializes the `firestore.Client` using the database ID from application settings.
*   **Collection References**: Provides centralized access to the following collections:
    *   `jobs`: Job processing data.
    *   `users`: User account information.
    *   `user_jobs`: Mapping between users and jobs.
    *   `reports`: Generated output or analysis reports.
*   **Utilities**: Includes a `now()` helper function for consistent UTC timestamps.

### Usage

Import the `db` client or specific collection references from `firestore_client.py` to perform database operations.

### TODO

*   Add data models or schemas for each collection to ensure type safety.
*   Implement repository patterns to decouple business logic from Firestore-specific syntax.
