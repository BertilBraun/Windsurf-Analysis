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

const DETAILED_PLAYBACK_AFTER_LAST_DETECTION_SEC = 1.0

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
        const hasCurrentTrackId = Object.prototype.hasOwnProperty.call(patch, 'currentTrackId')
        const hasIsPlaying = Object.prototype.hasOwnProperty.call(patch, 'isPlaying')
        return new PlayerState({
            mode: patch.mode ?? this.mode,
            currentTrackId: hasCurrentTrackId ? patch.currentTrackId! : this.currentTrackId,
            currentTimeSec: patch.currentTimeSec ?? this.currentTimeSec,
            playbackSpeed: patch.playbackSpeed ?? this.playbackSpeed,
            isPlaying: hasIsPlaying ? patch.isPlaying! : this.isPlaying,
            video: patch.video ?? this.video,
            tracks: patch.tracks ?? this.tracks,
            visibleTrackIds: patch.visibleTrackIds ?? this.visibleTrackIds,
            detectionTimesByTrack: patch.detectionTimesByTrack ?? this.detectionTimesByTrack,
        })
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
        const n = detectionTimes.length
        if (!n) return null
        const idx = this.binSearch(detectionTimes, timeSec)
        const i2 = Math.min(n - 1, idx)
        const i1 = Math.max(0, i2 - 1)
        if (i1 === i2) return detectionTimes[i1].detection
        const i0 = Math.max(0, i1 - 1)
        const i3 = Math.min(n - 1, i2 + 1)
        const P0 = detectionTimes[i0]
        const P1 = detectionTimes[i1]
        const P2 = detectionTimes[i2]
        const P3 = detectionTimes[i3]
        const x0 = P0.timeSec,
            x1 = P1.timeSec,
            x2 = P2.timeSec,
            x3 = P3.timeSec
        const bbox: [number, number, number, number] = [
            catmullRom1D(
                P0.detection.bbox[0],
                P1.detection.bbox[0],
                P2.detection.bbox[0],
                P3.detection.bbox[0],
                x0,
                x1,
                x2,
                x3,
                timeSec
            ),
            catmullRom1D(
                P0.detection.bbox[1],
                P1.detection.bbox[1],
                P2.detection.bbox[1],
                P3.detection.bbox[1],
                x0,
                x1,
                x2,
                x3,
                timeSec
            ),
            catmullRom1D(
                P0.detection.bbox[2],
                P1.detection.bbox[2],
                P2.detection.bbox[2],
                P3.detection.bbox[2],
                x0,
                x1,
                x2,
                x3,
                timeSec
            ),
            catmullRom1D(
                P0.detection.bbox[3],
                P1.detection.bbox[3],
                P2.detection.bbox[3],
                P3.detection.bbox[3],
                x0,
                x1,
                x2,
                x3,
                timeSec
            ),
        ]
        const confRaw = catmullRom1D(
            P0.detection.confidence,
            P1.detection.confidence,
            P2.detection.confidence,
            P3.detection.confidence,
            x0,
            x1,
            x2,
            x3,
            timeSec
        )
        const confidence = Math.max(0, Math.min(1, confRaw))
        const timePercent = Math.max(0, Math.min(1, timeSec / this.video.durationSeconds))
        return { bbox, confidence, time_percent: timePercent }
    }

    hasDetectionAfter(trackId: number, timeSec: number): boolean {
        const detectionTimes = this.detectionTimesByTrack.get(trackId) || []
        return detectionTimes[detectionTimes.length - 1].timeSec > timeSec - DETAILED_PLAYBACK_AFTER_LAST_DETECTION_SEC
    }

    binSearch(arr: Array<DetectionTime>, timeSec: number): number {
        let lo = 0
        let hi = arr.length - 1
        while (lo <= hi) {
            const mid = (lo + hi) >> 1
            const t = arr[mid].timeSec
            if (t < timeSec) lo = mid + 1
            else hi = mid - 1
        }
        return lo
    }
}

// Non-uniform Catmull-Rom spline interpolation for a scalar value over time (centripetal, alpha=0.5)
const catmullRom1D = (
    p0: number,
    p1: number,
    p2: number,
    p3: number,
    x0: number,
    x1: number,
    x2: number,
    x3: number,
    x: number
): number => {
    const seg = x2 - x1
    if (Math.abs(seg) < 1e-8) return p1
    const alpha = 0.5
    const tj = (ti: number, xi: number, xj: number) => ti + Math.pow(Math.abs(xj - xi), alpha)
    let t0 = 0
    let t1 = tj(t0, x0, x1)
    let t2 = tj(t1, x1, x2)
    let t3 = tj(t2, x2, x3)
    if (t1 <= t0) t1 = t0 + 1e-4
    if (t2 <= t1) t2 = t1 + 1e-4
    if (t3 <= t2) t3 = t2 + 1e-4
    const s = t1 + (t2 - t1) * ((x - x1) / (x2 - x1))
    const lerpParam = (a: number, b: number, ta: number, tb: number, t: number) => {
        const denom = tb - ta
        if (Math.abs(denom) < 1e-8) return a
        return ((tb - t) / denom) * a + ((t - ta) / denom) * b
    }
    const A1 = lerpParam(p0, p1, t0, t1, s)
    const A2 = lerpParam(p1, p2, t1, t2, s)
    const A3 = lerpParam(p2, p3, t2, t3, s)
    const B1 = lerpParam(A1, A2, t0, t2, s)
    const B2 = lerpParam(A2, A3, t1, t3, s)
    const C = lerpParam(B1, B2, t1, t2, s)
    return C
}
