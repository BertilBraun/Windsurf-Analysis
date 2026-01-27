Reusable React hooks for state management, data synchronization, and UI interactions.

### Data & Synchronization
* **useIngressScanner**: Monitors local directories for video files, performs stability checks to ensure files are finished copying, and manages uploads. Uses tab leadership to ensure only one tab performs scanning while broadcasting state to others.
* **useJobs**: Maintains a real-time list of processing jobs from Firestore. Hydrates job metadata with local file paths by matching SHA256 hashes from the local index.
* **useLocalFileIndex**: Scans a directory handle to index files by SHA256. Persists snapshots to IndexedDB and synchronizes updates across tabs.
* **useTabLeader**: Coordinates a single "leader" tab among multiple open instances using localStorage heartbeats. Prevents redundant background processing in non-leader tabs.

### State & Persistence
* **useSettings**: Manages and persists application-wide settings (upload quality, authentication credentials) in IndexedDB.
* **useOnce**: Tracks whether a specific action has been performed once, using IndexedDB for persistence across sessions.
* **useCappedValue**: Manages a numeric state automatically constrained within a defined min/max range.

### UI & Interaction
* **useTutorialController**: Manages the analyzer tutorial flow, including progress tracking, persistence, and automatic triggers for onboarding or contextual steps.
* **useZoom**: Manages zoom levels and 2D coordinate offsets for viewports. Supports focal-point zooming relative to specific coordinates (e.g., mouse cursor).
* **usePlaybackSpeed**: Manages video playback rates, cycling through a predefined set of speeds (0.25x, 0.5x, 1.0x, 2.0x, 4.0x, 8.0x).

### TODO
* Standardize error handling patterns across data-fetching hooks.
* Optimize useLocalFileIndex for very large directories to prevent UI lag during hashing.
