import { JobDetail, Track, TrackDetection, StabilizationTransform } from '../types'
import { DETAILED_PLAYBACK_AFTER_LAST_DETECTION_SEC } from './constants'

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

type PlayerStateInit = {
    mode: PlayerMode
    currentTrackId: number | null
    currentTimeSec: number
    playbackSpeed: number
    isPlaying: boolean
    video: VideoProperties
    tracks: Track[]
    stabilizationTransforms: StabilizationTransform[]
}

export class PlayerState {
    mode: PlayerMode
    currentTrackId: number | null
    currentTimeSec: number
    playbackSpeed: number
    isPlaying: boolean
    video: VideoProperties
    tracks: Track[]
    private stabilizationTransforms: StabilizationTransform[]
    private stabilizationCumulative: StabilizationTransform[]
    private visibleTrackIds: Set<number>

    constructor(params: PlayerStateInit) {
        this.mode = params.mode
        this.currentTrackId = params.currentTrackId
        this.currentTimeSec = params.currentTimeSec
        this.playbackSpeed = params.playbackSpeed
        this.isPlaying = params.isPlaying
        this.video = params.video
        this.tracks = params.tracks
        this.stabilizationTransforms = params.stabilizationTransforms.sort((a, b) => a.time_percent - b.time_percent)
        this.stabilizationCumulative = buildCumulativeInverse(this.stabilizationTransforms)
        this.visibleTrackIds = new Set(params.tracks.map(t => t.track_id))

        for (const track of this.tracks) {
            track.detections.sort((a, b) => a.time_percent - b.time_percent)
        }
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
            stabilizationTransforms: patch.stabilizationTransforms ?? this.stabilizationTransforms,
        })
    }

    static from(job: JobDetail, video: VideoProperties): PlayerState {
        return new PlayerState({
            mode: 'overview',
            currentTrackId: null,
            currentTimeSec: 0,
            playbackSpeed: 1.0,
            isPlaying: true,
            video,
            tracks: job.tracks,
            stabilizationTransforms: job.stabilization_transforms,
        })
    }

    interpolateDetectionByTime(trackId: number, timeSec: number): TrackDetection | null {
        const detectionTimes = this.tracks.find(t => t.track_id === trackId)?.detections || []
        const n = detectionTimes.length
        if (n < 5) return null

        const firstTime = this.time(detectionTimes[0]) - DETAILED_PLAYBACK_AFTER_LAST_DETECTION_SEC
        const lastTime = this.time(detectionTimes[n - 1]) + DETAILED_PLAYBACK_AFTER_LAST_DETECTION_SEC
        if (timeSec < firstTime || timeSec > lastTime) return null

        const idx = binSearch(detectionTimes, timeSec, t => this.time(t))
        const i2 = Math.min(n - 1, idx)
        const i1 = Math.max(0, i2 - 1)
        if (i1 === i2) return detectionTimes[i1]

        const alpha =
            (timeSec - this.time(detectionTimes[i1])) / (this.time(detectionTimes[i2]) - this.time(detectionTimes[i1]))
        const [x10, x11, x12, x13] = detectionTimes[i1].bbox
        const [x20, x21, x22, x23] = detectionTimes[i2].bbox
        const c1 = detectionTimes[i1].confidence
        const c2 = detectionTimes[i2].confidence

        return {
            bbox: [
                x10 + alpha * (x20 - x10),
                x11 + alpha * (x21 - x11),
                x12 + alpha * (x22 - x12),
                x13 + alpha * (x23 - x13),
            ],
            confidence: c1 + alpha * (c2 - c1),
            time_percent: timeSec / this.video.durationSeconds,
        }

        const i0 = Math.max(0, i1 - 1)
        const i3 = Math.min(n - 1, i2 + 1)
        const P0 = detectionTimes[i0]
        const P1 = detectionTimes[i1]
        const P2 = detectionTimes[i2]
        const P3 = detectionTimes[i3]
        const x0 = this.time(P0)
        const x1 = this.time(P1)
        const x2 = this.time(P2)
        const x3 = this.time(P3)
        const bbox: [number, number, number, number] = [
            catmullRom1D(P0.bbox[0], P1.bbox[0], P2.bbox[0], P3.bbox[0], x0, x1, x2, x3, timeSec),
            catmullRom1D(P0.bbox[1], P1.bbox[1], P2.bbox[1], P3.bbox[1], x0, x1, x2, x3, timeSec),
            catmullRom1D(P0.bbox[2], P1.bbox[2], P2.bbox[2], P3.bbox[2], x0, x1, x2, x3, timeSec),
            catmullRom1D(P0.bbox[3], P1.bbox[3], P2.bbox[3], P3.bbox[3], x0, x1, x2, x3, timeSec),
        ]
        const confRaw = catmullRom1D(
            P0.confidence,
            P1.confidence,
            P2.confidence,
            P3.confidence,
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
        const detectionTimes = this.tracks.find(t => t.track_id === trackId)?.detections || []
        return (
            this.time(detectionTimes[detectionTimes.length - 1]) + DETAILED_PLAYBACK_AFTER_LAST_DETECTION_SEC >= timeSec
        )
    }

    getStabilizationAt(timeSec: number): { dx: number; dy: number; da: number } {
        const arr = this.stabilizationCumulative
        const n = arr.length
        if (!n) return { dx: 0, dy: 0, da: 0 }
        // indices for Catmull–Rom interpolation
        const idx = binSearch(arr, timeSec, t => this.time(t))
        const i2 = Math.min(n - 1, idx)
        // Temporary: nearest-sample; interpolation below can be re-enabled if needed
        return { dx: arr[i2].dx, dy: arr[i2].dy, da: arr[i2].da }

        const i1 = Math.max(0, i2 - 1)
        if (i1 === i2) return { dx: arr[i1].dx, dy: arr[i1].dy, da: arr[i1].da }
        const i0 = Math.max(0, i1 - 1)
        const i3 = Math.min(n - 1, i2 + 1)
        const P0 = arr[i0]
        const P1 = arr[i1]
        const P2 = arr[i2]
        const P3 = arr[i3]
        const x0 = this.time(P0)
        const x1 = this.time(P1)
        const x2 = this.time(P2)
        const x3 = this.time(P3)
        const dx = catmullRom1D(P0.dx, P1.dx, P2.dx, P3.dx, x0, x1, x2, x3, timeSec)
        const dy = catmullRom1D(P0.dy, P1.dy, P2.dy, P3.dy, x0, x1, x2, x3, timeSec)
        const da = catmullRom1D(P0.da, P1.da, P2.da, P3.da, x0, x1, x2, x3, timeSec)
        return { dx, dy, da }
    }

    time(t: { time_percent: number }): number {
        return t.time_percent * this.video.durationSeconds
    }
}

function buildCumulativeInverse(arr: StabilizationTransform[]): StabilizationTransform[] {
    if (!arr || arr.length === 0) return []
    const out: StabilizationTransform[] = []
    let cumAngle = 0
    let cumTx = 0
    let cumTy = 0
    for (const t of arr) {
        const ca = Math.cos(t.da)
        const sa = Math.sin(t.da)
        // Update cumulative camera motion: C_total = H_delta * C_total
        const newTx = ca * cumTx - sa * cumTy + t.dx
        const newTy = sa * cumTx + ca * cumTy + t.dy
        cumTx = newTx
        cumTy = newTy
        cumAngle += t.da

        // Inverse to stabilize: R_inv = R_total^T, T_inv = -R_total^T * T_total
        const c = Math.cos(cumAngle)
        const s = Math.sin(cumAngle)
        const invDx = -(c * cumTx + s * cumTy)
        const invDy = -(-s * cumTx + c * cumTy)
        const invAngle = -cumAngle
        out.push({ time_percent: t.time_percent, dx: invDx, dy: invDy, da: invAngle })
    }
    return out
}

// Non-uniform Catmull-Rom spline interpolation for a scalar value over time (centripetal, alpha=0.5)
const ALPHA = 0.5

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
    if (Math.abs(x2 - x1) < 1e-8) return p1

    function tj(ti: number, xi: number, xj: number): number {
        return ti + Math.pow(Math.abs(xj - xi), ALPHA)
    }
    function lerpParam(a: number, b: number, ta: number, tb: number, t: number): number {
        const denom = tb - ta
        if (Math.abs(denom) < 1e-8) return a
        return ((tb - t) / denom) * a + ((t - ta) / denom) * b
    }

    let t0 = 0
    let t1 = tj(t0, x0, x1)
    let t2 = tj(t1, x1, x2)
    let t3 = tj(t2, x2, x3)
    if (t1 <= t0) t1 = t0 + 1e-4
    if (t2 <= t1) t2 = t1 + 1e-4
    if (t3 <= t2) t3 = t2 + 1e-4

    const s = t1 + (t2 - t1) * ((x - x1) / (x2 - x1))

    const A1 = lerpParam(p0, p1, t0, t1, s)
    const A2 = lerpParam(p1, p2, t1, t2, s)
    const A3 = lerpParam(p2, p3, t2, t3, s)

    const B1 = lerpParam(A1, A2, t0, t2, s)
    const B2 = lerpParam(A2, A3, t1, t3, s)

    return lerpParam(B1, B2, t1, t2, s)
}

function binSearch<T>(arr: Array<T>, value: number, accessor: (t: T) => number): number {
    /* Performs a binary search on an array of objects, using a provided accessor function to get the value to search for.
    Returns the index of the element in the array that is the closest to the given value. */
    let lo = 0
    let hi = arr.length - 1
    while (lo <= hi) {
        const mid = (lo + hi) >> 1
        const t = accessor(arr[mid])
        if (t < value) lo = mid + 1
        else hi = mid - 1
    }
    return lo
}
