import { JobDetail, Track, TrackDetection, StabilizationTransform } from '../types'
import { assert } from '../utils/assert'

export type PlayerMode = 'overview' | 'detailed'

type PlayerStateInit = {
    mode: PlayerMode
    currentTrackId: number | null
    frameCount: number
    tracks: Track[]
    stabilizationByFrame: Array<{ dx: number; dy: number; da: number }>
}

export class PlayerState {
    mode: PlayerMode
    currentTrackId: number | null
    frameCount: number // this is the total frame count of the video - it will always be >0
    tracks: Track[]

    private stabilizationByFrame: Array<{ dx: number; dy: number; da: number }>

    constructor(params: PlayerStateInit) {
        this.mode = params.mode
        this.currentTrackId = params.currentTrackId
        this.frameCount = params.frameCount
        this.tracks = params.tracks
        this.stabilizationByFrame = params.stabilizationByFrame
    }

    copy(patch: Partial<PlayerStateInit>): PlayerState {
        const hasCurrentTrackId = Object.prototype.hasOwnProperty.call(patch, 'currentTrackId')
        return new PlayerState({
            mode: patch.mode ?? this.mode,
            currentTrackId: hasCurrentTrackId ? patch.currentTrackId! : this.currentTrackId,
            frameCount: patch.frameCount ?? this.frameCount,
            tracks: patch.tracks ?? this.tracks,
            stabilizationByFrame: patch.stabilizationByFrame ?? this.stabilizationByFrame,
        })
    }

    static from(job: JobDetail, frameCount: number): PlayerState {
        assert(frameCount > 0, 'Invalid frame count')

        const tracks = job.tracks.map(track => ({
            ...track,
            detections: [...track.detections].sort((a, b) => a.time_percent - b.time_percent),
        }))

        return new PlayerState({
            mode: 'overview',
            currentTrackId: null,
            frameCount,
            tracks,
            stabilizationByFrame: materializeStabilization(job.stabilization_transforms, frameCount),
        })
    }

    getStabilizationAtFrame(frameIndex: number): { dx: number; dy: number; da: number } {
        assert(frameIndex >= 0 && frameIndex < this.frameCount, 'Invalid frame index')
        return this.stabilizationByFrame[frameIndex]
    }

    getDetectionAtFrame(trackId: number, frameIndex: number): TrackDetection | null {
        assert(frameIndex >= 0 && frameIndex < this.frameCount, 'Invalid frame index')
        const track = this.getTrackById(trackId)
        const detections = track.detections
        // Binary search for the detection at the frame index where frameIndexForPercent(detection.time_percent) equals frameIndex
        let lo = 0
        let hi = detections.length - 1
        while (lo <= hi) {
            const mid = Math.floor((lo + hi) / 2)
            const midFrameIndex = this.frameIndexForPercent(detections[mid]!.time_percent)
            if (midFrameIndex === frameIndex) return detections[mid]
            if (midFrameIndex < frameIndex) lo = mid + 1
            else hi = mid - 1
        }
        return null
    }

    getDetectionAtFrameRequired(trackId: number, frameIndex: number): TrackDetection {
        const det = this.getDetectionAtFrame(trackId, frameIndex)
        if (!det) throw new Error(`Missing detection for track ${trackId} at frame ${frameIndex}.`)
        return det
    }

    getTrackFrameRange(trackId: number): { startFrameIndex: number; endFrameIndex: number } {
        const track = this.getTrackById(trackId)
        const startFrameIndex = this.frameIndexForPercent(track.start_percent)
        const endFrameIndex = this.frameIndexForPercent(track.end_percent)
        return { startFrameIndex, endFrameIndex }
    }

    isTrackActiveAtFrame(trackId: number, frameIndex: number): boolean {
        const r = this.getTrackFrameRange(trackId)
        if (!r) return false
        return frameIndex >= r.startFrameIndex && frameIndex <= r.endFrameIndex
    }

    getTrackById(trackId: number): Track {
        const track = this.tracks.find(track => track.track_id === trackId)
        assert(track !== undefined, 'Track not found for id ${trackId}')
        return track!
    }

    frameIndexForPercent(percent: number): number {
        assert(percent >= 0 && percent <= 1, 'Invalid percent')
        return clampInt(Math.round(percent * this.frameCount), 0, this.frameCount - 1)
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
        const idx = clampInt(Math.round(t.time_percent * frameCount), 0, frameCount - 1)
        out[idx] = { dx: t.dx, dy: t.dy, da: t.da }
    }

    for (const t of out) {
        assert(t !== null, 'Missing stabilization!!!')
    }

    return out as Array<{ dx: number; dy: number; da: number }>
}
