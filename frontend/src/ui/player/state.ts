import { JobDetail, Track, TrackDetection, StabilizationTransform } from '../types'

export type VideoProperties = {
    width: number
    height: number
    frameCount: number
}

export type PlayerMode = 'overview' | 'detailed'

type PlayerStateInit = {
    mode: PlayerMode
    currentTrackId: number | null
    isPlaying: boolean
    video: VideoProperties
    tracks: Track[]
    stabilizationByFrame: Array<{ dx: number; dy: number; da: number }>
    detectionsByTrackId: Map<number, Array<TrackDetection | null>>
}

export class PlayerState {
    mode: PlayerMode
    currentTrackId: number | null
    isPlaying: boolean
    video: VideoProperties
    tracks: Track[]

    private stabilizationByFrame: Array<{ dx: number; dy: number; da: number }>
    private detectionsByTrackId: Map<number, Array<TrackDetection | null>>

    constructor(params: PlayerStateInit) {
        this.mode = params.mode
        this.currentTrackId = params.currentTrackId
        this.isPlaying = params.isPlaying
        this.video = params.video
        this.tracks = params.tracks
        this.stabilizationByFrame = params.stabilizationByFrame
        this.detectionsByTrackId = params.detectionsByTrackId
    }

    copy(patch: Partial<PlayerStateInit>): PlayerState {
        const hasCurrentTrackId = Object.prototype.hasOwnProperty.call(patch, 'currentTrackId')
        const hasIsPlaying = Object.prototype.hasOwnProperty.call(patch, 'isPlaying')
        return new PlayerState({
            mode: patch.mode ?? this.mode,
            currentTrackId: hasCurrentTrackId ? patch.currentTrackId! : this.currentTrackId,
            isPlaying: hasIsPlaying ? patch.isPlaying! : this.isPlaying,
            video: patch.video ?? this.video,
            tracks: patch.tracks ?? this.tracks,
            stabilizationByFrame: patch.stabilizationByFrame ?? this.stabilizationByFrame,
            detectionsByTrackId: patch.detectionsByTrackId ?? this.detectionsByTrackId,
        })
    }

    static from(job: JobDetail, video: VideoProperties): PlayerState {
        const frameCount = Math.max(0, video.frameCount | 0)

        const tracks = job.tracks.map(t => ({
            ...t,
            detections: [...(t.detections ?? [])].sort((a, b) => a.time_percent - b.time_percent),
        }))

        return new PlayerState({
            mode: 'overview',
            currentTrackId: null,
            isPlaying: true,
            video,
            tracks,
            stabilizationByFrame: materializeStabilization(job.stabilization_transforms ?? [], frameCount),
            detectionsByTrackId: materializeDetectionsByTrackId(tracks, frameCount),
        })
    }

    getStabilizationAtFrame(frameIndex: number): { dx: number; dy: number; da: number } {
        const n = this.video.frameCount
        if (n <= 0) return { dx: 0, dy: 0, da: 0 }
        const idx = clampInt(frameIndex, 0, n - 1)
        return this.stabilizationByFrame[idx] ?? { dx: 0, dy: 0, da: 0 }
    }

    getDetectionAtFrame(trackId: number, frameIndex: number): TrackDetection | null {
        const arr = this.detectionsByTrackId.get(trackId)
        if (!arr || arr.length === 0) return null
        const idx = clampInt(frameIndex, 0, arr.length - 1)
        return arr[idx] ?? null
    }

    getTrackFrameRange(trackId: number): { startFrameIndex: number; endFrameIndex: number } | null {
        const t = this.tracks.find(t0 => t0.track_id === trackId)
        if (!t) return null
        const n = this.video.frameCount
        if (n <= 0) return null
        const startFrameIndex = clampInt(Math.round(t.start_percent * (n - 1)), 0, n - 1)
        const endFrameIndex = clampInt(Math.round(t.end_percent * (n - 1)), 0, n - 1)
        return { startFrameIndex, endFrameIndex }
    }

    isTrackActiveAtFrame(trackId: number, frameIndex: number): boolean {
        const r = this.getTrackFrameRange(trackId)
        if (!r) return false
        return frameIndex >= r.startFrameIndex && frameIndex <= r.endFrameIndex
    }
}

function clampInt(x: number, lo: number, hi: number) {
    const xi = x | 0
    return Math.max(lo, Math.min(hi, xi))
}

function materializeStabilization(
    transforms: StabilizationTransform[],
    frameCount: number
): Array<{ dx: number; dy: number; da: number }> {
    if (frameCount <= 0) return []
    if (transforms.length === 0) return new Array(frameCount).fill(0).map(() => ({ dx: 0, dy: 0, da: 0 }))

    const sorted = [...transforms].sort((a, b) => a.time_percent - b.time_percent)

    if (sorted.length === frameCount) {
        return sorted.map(t => ({ dx: t.dx, dy: t.dy, da: t.da }))
    }

    const out: Array<{ dx: number; dy: number; da: number } | null> = new Array(frameCount).fill(null)
    for (const t of sorted) {
        const idx = clampInt(Math.round(t.time_percent * (frameCount - 1)), 0, frameCount - 1)
        out[idx] = { dx: t.dx, dy: t.dy, da: t.da }
    }

    let last: { dx: number; dy: number; da: number } = out[0] ?? { dx: 0, dy: 0, da: 0 }
    for (let i = 0; i < frameCount; i++) {
        if (out[i]) last = out[i]!
        else out[i] = last
    }
    return out as Array<{ dx: number; dy: number; da: number }>
}

function materializeDetectionsByTrackId(tracks: Track[], frameCount: number): Map<number, Array<TrackDetection | null>> {
    const out = new Map<number, Array<TrackDetection | null>>()
    if (frameCount <= 0) return out

    for (const t of tracks) {
        const dets = t.detections ?? []
        const arr: Array<TrackDetection | null> = new Array(frameCount).fill(null)

        if (dets.length === frameCount) {
            for (let i = 0; i < frameCount; i++) arr[i] = dets[i] ?? null
        } else {
            for (const d of dets) {
                const idx = clampInt(Math.round(d.time_percent * (frameCount - 1)), 0, frameCount - 1)
                arr[idx] = d
            }
        }

        out.set(t.track_id, arr)
    }
    return out
}
