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
    results_json?: Record<string, unknown> | null
}

export type ReportType = 'missed_detection' | 'false_association' | 'other'



