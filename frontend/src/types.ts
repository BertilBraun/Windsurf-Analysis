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

export type JobSummary = {
    id: string
    status: JobStatus
    created_at: string
    updated_at: string
    original_checksum_sha256: string
    dominant_orientation: number
}

export type JobDetail = JobSummary & {
    tracks: any[]
    stabilization_transforms: any[]
}
