import { JobDetail, Track, TrackDetection } from '../types'

export type VideoProperties = {
    width: number
    height: number
    durationSeconds: number
}

export type PlayerMode = 'overview' | 'detailed'

export type DetectionTime = {
    timeSec: number
    detection: TrackDetection
}

const lerp = (a: number, b: number, t: number) => a * (1 - t) + b * t

type PlayerStateInit = {
    mode: PlayerMode
    currentTrackId: number | null
    currentTimeSec: number
    playbackSpeed: number
    isPlaying: boolean
    video: VideoProperties
    tracks: Track[]
    visibleTrackIds: Set<number>
    detectionTimesByTrack: Map<number, Array<DetectionTime>>
}

export class PlayerState {
    mode: PlayerMode
    currentTrackId: number | null
    currentTimeSec: number
    playbackSpeed: number
    isPlaying: boolean
    video: VideoProperties
    tracks: Track[]
    visibleTrackIds: Set<number>
    // per track sorted by time seconds for fast nearest lookup
    detectionTimesByTrack: Map<number, Array<DetectionTime>>

    constructor(params: PlayerStateInit) {
        this.mode = params.mode
        this.currentTrackId = params.currentTrackId
        this.currentTimeSec = params.currentTimeSec
        this.playbackSpeed = params.playbackSpeed
        this.isPlaying = params.isPlaying
        this.video = params.video
        this.tracks = params.tracks
        this.visibleTrackIds = params.visibleTrackIds
        this.detectionTimesByTrack = params.detectionTimesByTrack
    }

    copy(patch: Partial<PlayerStateInit>): PlayerState {
        return new PlayerState({
            mode: patch.mode ?? this.mode,
            currentTrackId: patch.currentTrackId ?? this.currentTrackId,
            currentTimeSec: patch.currentTimeSec ?? this.currentTimeSec,
            playbackSpeed: patch.playbackSpeed ?? this.playbackSpeed,
            isPlaying: patch.isPlaying ?? this.isPlaying,
            video: patch.video ?? this.video,
            tracks: patch.tracks ?? this.tracks,
            visibleTrackIds: patch.visibleTrackIds ?? this.visibleTrackIds,
            detectionTimesByTrack: patch.detectionTimesByTrack ?? this.detectionTimesByTrack,
        })
    }

    togglePlay(): PlayerState {
        return this.copy({ isPlaying: !this.isPlaying })
    }

    static from(job: JobDetail, video: VideoProperties): PlayerState {
        const tracks = job.tracks || []
        const visibleTrackIds = new Set<number>(tracks.map(t => t.track_id))
        const detectionTimesByTrack = new Map<number, Array<DetectionTime>>()

        for (const t of tracks) {
            const arrTimes: Array<DetectionTime> = []
            for (const det of t.detections) {
                const timeSec = det.time_percent * video.durationSeconds
                arrTimes.push({ timeSec, detection: det })
            }
            arrTimes.sort((a, b) => a.timeSec - b.timeSec)
            detectionTimesByTrack.set(t.track_id, arrTimes)
        }

        return new PlayerState({
            mode: 'overview',
            currentTrackId: null,
            currentTimeSec: 0,
            playbackSpeed: 1.0,
            isPlaying: false,
            video,
            tracks,
            visibleTrackIds,
            detectionTimesByTrack,
        })
    }

    interpolateDetectionByTime(trackId: number, timeSec: number): TrackDetection | null {
        const detectionTimes = this.detectionTimesByTrack.get(trackId) || []
        if (!detectionTimes.length) return null
        let lo = 0
        let hi = detectionTimes.length - 1
        while (lo < hi) {
            const mid = (lo + hi) >> 1
            const t = detectionTimes[mid].timeSec
            if (t < timeSec) lo = mid + 1
            else hi = mid
        }
        const pred = detectionTimes[Math.max(0, lo - 1)]
        const succ = detectionTimes[Math.min(detectionTimes.length - 1, lo)]

        const dt = succ.timeSec - pred.timeSec
        const t = (timeSec - pred.timeSec) / dt

        const bbox: [number, number, number, number] = [
            lerp(pred.detection.bbox[0], succ.detection.bbox[0], t),
            lerp(pred.detection.bbox[1], succ.detection.bbox[1], t),
            lerp(pred.detection.bbox[2], succ.detection.bbox[2], t),
            lerp(pred.detection.bbox[3], succ.detection.bbox[3], t),
        ]
        const confidence = lerp(pred.detection.confidence, succ.detection.confidence, t)
        const timePercent = lerp(pred.detection.time_percent, succ.detection.time_percent, t)

        return {
            bbox,
            confidence,
            time_percent: timePercent,
        }
    }

    hasDetectionAfter(trackId: number, timeSec: number): boolean {
        const detectionTimes = this.detectionTimesByTrack.get(trackId) || []
        if (!detectionTimes.length) return false
        let lo = 0
        let hi = detectionTimes.length - 1
        while (lo < hi) {
            const mid = (lo + hi) >> 1
            const t = detectionTimes[mid].timeSec
            if (t < timeSec) lo = mid + 1
            else hi = mid
        }
        return lo < detectionTimes.length && detectionTimes[lo].timeSec > timeSec
    }

    findLatestDetectionAtOrBeforeTime(trackId: number, timeSec: number, maxAgeSec: number): DetectionTime | null {
        const detectionTimes = this.detectionTimesByTrack.get(trackId) || []
        if (!detectionTimes.length) return null
        let lo = 0
        let hi = detectionTimes.length - 1
        while (lo <= hi) {
            const mid = (lo + hi) >> 1
            const t = detectionTimes[mid].timeSec
            if (t <= timeSec) lo = mid + 1
            else hi = mid - 1
        }
        const idx = Math.max(-1, Math.min(detectionTimes.length - 1, hi))
        if (idx < 0) return null
        const cand = detectionTimes[idx]
        if (timeSec - cand.timeSec <= maxAgeSec) return cand
        return null
    }
}
