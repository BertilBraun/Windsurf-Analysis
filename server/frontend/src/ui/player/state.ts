import { JobDetail, Track, TrackDetection } from '../types'

export type VideoProperties = {
    width: number
    height: number
    durationSeconds: number
}

export type PlayerMode = 'overview' | 'detailed'

export type PlayerState = {
    mode: PlayerMode
    currentTrackId: number | null
    currentTimeSec: number
    playbackSpeed: number
    isPlaying: boolean
    video: VideoProperties
    tracks: Track[]
    visibleTrackIds: Set<number>
    // per track sorted by time seconds for fast nearest lookup
    detectionTimesByTrack: Map<number, Array<{ timeSec: number; detection: TrackDetection }>>
}

export function buildPlayerState(job: JobDetail, video: VideoProperties): PlayerState {
    const tracks = job.tracks || []
    const visibleTrackIds = new Set<number>(tracks.map(t => t.track_id))
    const detectionTimesByTrack = new Map<number, Array<{ timeSec: number; detection: TrackDetection }>>()

    for (const t of tracks) {
        const arrTimes: Array<{ timeSec: number; detection: TrackDetection }> = []
        for (const det of t.detections) {
            const timeSec = det.time_percent * video.durationSeconds
            arrTimes.push({ timeSec, detection: det })
        }
        arrTimes.sort((a, b) => a.timeSec - b.timeSec)
        detectionTimesByTrack.set(t.track_id, arrTimes)
    }

    return {
        mode: 'overview',
        currentTrackId: null,
        currentTimeSec: 0,
        playbackSpeed: 1.0,
        isPlaying: false,
        video,
        tracks,
        visibleTrackIds,
        detectionTimesByTrack,
    }
}

export function findNearestDetectionByTime(
    detectionTimes: Array<{ timeSec: number; detection: TrackDetection }>,
    timeSec: number,
    maxDeltaSec: number
): { timeSec: number; detection: TrackDetection } | null {
    if (!detectionTimes.length) return null
    let lo = 0
    let hi = detectionTimes.length - 1
    while (lo <= hi) {
        const mid = (lo + hi) >> 1
        const t = detectionTimes[mid].timeSec
        if (t === timeSec) return detectionTimes[mid]
        if (t < timeSec) lo = mid + 1
        else hi = mid - 1
    }
    const candidates: Array<{ timeSec: number; detection: TrackDetection }> = []
    if (lo < detectionTimes.length) candidates.push(detectionTimes[lo])
    if (lo < detectionTimes.length - 1) candidates.push(detectionTimes[lo + 1])
    if (lo >= 1) candidates.push(detectionTimes[lo - 1])
    if (lo >= 2) candidates.push(detectionTimes[lo - 2])
    let best: { timeSec: number; detection: TrackDetection } | null = null
    let bestDelta = Infinity
    for (const c of candidates) {
        const d = Math.abs(c.timeSec - timeSec)
        if (d <= maxDeltaSec && d < bestDelta) {
            best = c
            bestDelta = d
        }
    }
    return best
}

export function hasDetectionNearTime(
    trackId: number,
    timeSec: number,
    state: PlayerState,
    maxDeltaSec: number
): boolean {
    const arr = state.detectionTimesByTrack.get(trackId) || []
    return findNearestDetectionByTime(arr, timeSec, maxDeltaSec) != null
}
