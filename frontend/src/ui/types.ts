export type JobStatus =
    | 'pending'
    | 'starting'
    | 'orientation'
    | 'stabilization'
    | 'detection'
    | 'appearance'
    | 'tracking'
    | 'succeeded'
    | 'failed'
    | 'canceled'

export type JobSummary = {
    id: string
    status: JobStatus
    created_at: string
    updated_at: string
    original_checksum_sha256: string
    dominant_orientation: number
    local_relative_path?: string | null
}

export type JobDetail = JobSummary & {
    tracks: Track[]
    stabilization_transforms: StabilizationTransform[]
}

export type TrackDetection = {
    time_percent: number
    bbox: [number, number, number, number]
    confidence: number
}

export type Track = {
    track_id: number
    start_percent: number
    end_percent: number
    start_time_seconds: number
    duration_seconds: number
    detections: TrackDetection[]
}

export type ReportType = 'missed_detection' | 'false_association' | 'other'

export type UploadQuality = 'original' | 'high' | 'medium' | 'minimum'

export type StabilizationTransform = {
    time_percent: number
    dx: number
    dy: number
    da: number // radians
}
