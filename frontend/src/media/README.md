# Media Utilities

Utilities for extracting and indexing video packet metadata.

### Features

*   **Metadata Extraction**: Uses `mediabunny` to extract packet metadata (timestamp, duration, keyframe status, and sequence number) from video tracks.
*   **Packet Sorting**: Automatically sorts extracted metadata by presentation timestamp (PTS) and sequence number.
*   **Timestamp Indexing**: Provides a binary search utility to find the closest packet index for a given target time.

### Key Types

*   **VideoPacketMeta**: Represents encoded video packet properties:
    *   `ts`: Presentation timestamp.
    *   `dur`: Packet duration.
    *   `key`: Boolean indicating if the packet is a keyframe.
    *   `tie`: Sequence number used to break ties for identical timestamps.

### TODO

*   Add error handling for specific `mediabunny` failure modes.
*   Support for audio packet metadata extraction.
