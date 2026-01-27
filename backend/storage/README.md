# Storage Utilities

Utilities for interacting with cloud storage services, specifically focused on JSON data persistence.

## Google Cloud Storage (GCS)
The `gcs_json.py` module provides helpers for managing JSON objects in GCS:

*   **Authentication**: Initializes a GCS client using the `GCP_SERVICE_ACCOUNT_JSON` environment variable (JSON string). Falls back to default credentials if the variable is unset.
*   **JSON Upload**: `upload_json` serializes dictionaries to GCS with `no-store` cache control and UTF-8 encoding.
*   **JSON Download**: `download_json` retrieves and parses JSON objects; returns `None` if the object does not exist.
*   **Path Validation**: Enforces safe object names by blocking directory traversal (e.g., `..`) and leading slashes.

## TODO
*   Add support for non-JSON file formats (e.g., raw bytes or CSV).
*   Implement batch upload/download capabilities.
