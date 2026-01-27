# Services

Business logic layer responsible for orchestrating data between controllers, models, and external processing workers.

### Key Responsibilities
* **Business Logic:** Encapsulates core rules and processing workflows.
* **Workflow Management:** Orchestrates complex tasks such as transcoding and thumbnail generation.
* **Integration:** Interfaces with external storage providers and message queues.
* **Decoupling:** Separates API endpoints from data access and underlying processing implementations.

### TODO
* Implement `VideoService` for handling video uploads and tracking processing status.
* Implement `StorageService` for managing file persistence and retrieval.
* Integrate error handling and logging middleware.
