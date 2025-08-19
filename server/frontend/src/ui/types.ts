export type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'canceled'

export type JobSummary = {
    id: string
    video_id: string
    model: string
    status: JobStatus
    created_at: string
    updated_at: string
}

export type JobDetail = JobSummary & {
    original_file_path: string
    original_checksum_sha256: string
    tracks?: Track[] | null
    dominant_orientation?: number | null
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
