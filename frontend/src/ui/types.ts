/**
 * UI-related type definitions for job management, tracking, and reporting.
 * @module
 */

/**
 * Represents the current lifecycle state of a processing job.
 */
export type JobStatus =
    | 'uploading'
    | 'starting'
    | 'orientation'
    | 'stabilization'
    | 'detection'
    | 'appearance'
    | 'tracking'
    | 'succeeded'
    | 'failed'
    | 'canceled'

/**
 * Basic metadata and status information for a processing job.
 */
export type JobSummary = {
    /** Unique identifier for the job. */
    id: string
    /** Current processing status. */
    status: JobStatus
    /** ISO 8601 timestamp of creation. */
    created_at: string
    /** ISO 8601 timestamp of last update. */
    updated_at: string
    /** SHA-256 hash of the source file. */
    sha256: string
    /** Detected dominant orientation in degrees (e.g., 0, 90, 180, 270). */
    dominant_orientation: number
    /** Path relative to the local storage root. */
    local_relative_path?: string | null
    /**
     * List of local file paths associated with the job's checksum.
     * Derived from local IndexedDB sha-to-paths mapping.
     */
    local_relative_paths?: string[] | null
    /** Most recently accessed local path for the file. */
    last_known_local_path?: string | null
}

/**
 * Detailed job information including tracking data and stabilization transforms.
 */
export type JobDetail = JobSummary & {
    /** List of detected object tracks. */
    tracks: Track[]
    /** Frame-by-frame stabilization data. */
    stabilization_transforms: StabilizationTransform[]
}

/**
 * Spatial and temporal data for a single detection within a track.
 */
export type TrackDetection = {
    /** Position in the video as a percentage (0.0 to 1.0). */
    time_percent: number
    /** Bounding box coordinates [x, y, width, height] as percentages (0.0 to 1.0). */
    bbox: [number, number, number, number]
    /** Normalized anchor point [x, y] for the detection. */
    anchor: [number, number]
    /** Scale factor of the detection. */
    scale: number
    /** Confidence score of the detection (0.0 to 1.0). */
    confidence: number
    /** Whether the detection was interpolated between keyframes. */
    interpolated: boolean
}

/**
 * A sequence of detections representing a single object tracked over time.
 */
export type Track = {
    /** Unique identifier for the track within the job. */
    track_id: number
    /** Start position in the video as a percentage (0.0 to 1.0). */
    start_percent: number
    /** End position in the video as a percentage (0.0 to 1.0). */
    end_percent: number
    /** Start time in seconds from the beginning of the video. */
    start_time_seconds: number
    /** Total duration of the track in seconds. */
    duration_seconds: number
    /** Array of detection points belonging to this track. */
    detections: TrackDetection[]
}

/**
 * Categories for user-submitted reports or feedback.
 */
export type ReportType = 'missed_detection' | 'false_association' | 'visual_problem' | 'feedback' | 'other'

/**
 * Quality presets for video file uploads.
 */
export type UploadQuality = 'original' | 'high' | 'medium' | 'minimum'

/**
 * Geometric transformation applied to a frame for stabilization.
 */
export type StabilizationTransform = {
    /** Position in the video as a percentage (0.0 to 1.0). */
    time_percent: number
    /** Horizontal translation. */
    dx: number
    /** Vertical translation. */
    dy: number
    /** Rotation in radians. */
    da: number
}
