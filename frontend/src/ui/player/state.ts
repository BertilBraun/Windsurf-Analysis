/**
 * State management for the video player.
 * Handles track data, stabilization transforms, and frame-based lookups.
 */

import { JobDetail, Track, TrackDetection } from '../types'
import { assert } from '../utils/assert'
import { frameIndexForPercent, getClosestDetectionAtFrame } from './trackMath'

/**
 * Represents the viewing mode of the player.
 * 'overview' shows all tracks, while 'detailed' focuses on a specific track.
 */
export type PlayerMode = 'overview' | 'detailed'

type PlayerStateInit = {
    mode: PlayerMode
    currentTrackId: number | null
    frameCount: number
    tracks: Track[]
    stabilizationByFrame: Array<{ dx: number; dy: number; da: number }>
}

/**
 * Manages the state of the video player, including track data and stabilization transforms.
 */
export class PlayerState {
    /** Current viewing mode. */
    mode: PlayerMode
    /** ID of the currently selected track, or null if none. */
    currentTrackId: number | null
    /** Total number of frames in the video (always > 0). */
    frameCount: number
    /** List of tracks associated with the video. */
    tracks: Track[]

    private stabilizationByFrame: Array<{ dx: number; dy: number; da: number }>

    /**
     * @param params - Initial state parameters.
     */
    constructor(params: PlayerStateInit) {
        this.mode = params.mode
        this.currentTrackId = params.currentTrackId
        this.frameCount = params.frameCount
        this.tracks = params.tracks
        this.stabilizationByFrame = params.stabilizationByFrame
    }

    /**
     * Creates a copy of the state with optional property overrides.
     * @param patch - Properties to update.
     * @returns A new PlayerState instance.
     */
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

    /**
     * Initializes PlayerState from a job detail and frame count.
     * Ensures tracks and stabilization data are sorted and validated.
     * @param job - The job detail containing tracks and stabilization data.
     * @param frameCount - Total number of frames in the video.
     * @returns A new PlayerState instance.
     */
    static from(job: JobDetail, frameCount: number): PlayerState {
        assert(frameCount > 0, 'Invalid frame count')

        const tracks = job.tracks.map(track => ({
            ...track,
            detections: [...track.detections].sort((a, b) => a.time_percent - b.time_percent),
        }))

        const stabilizationByFrame = [...job.stabilization_transforms].sort((a, b) => a.time_percent - b.time_percent).map(t => ({ dx: t.dx, dy: t.dy, da: t.da }))

        assert(stabilizationByFrame.length === frameCount, 'Missing Stabilization!!!')

        return new PlayerState({
            mode: 'overview',
            currentTrackId: null,
            frameCount,
            tracks,
            stabilizationByFrame
        })
    }

    /**
     * Gets the stabilization transform (dx, dy, da) for a specific frame index.
     * @param frameIndex - The index of the frame.
     * @returns The transform object.
     */
    getStabilizationAtFrame(frameIndex: number): { dx: number; dy: number; da: number } {
        assert(frameIndex >= 0 && frameIndex < this.frameCount, 'Invalid frame index')
        return this.stabilizationByFrame[frameIndex]!
    }

    /**
     * Retrieves the detection for a track at a specific frame index, if it exists.
     * Uses binary search for efficiency.
     * @param trackId - The ID of the track.
     * @param frameIndex - The index of the frame.
     * @returns The detection or null if not found.
     */
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

    /**
     * Finds the detection in a track closest to the specified frame index.
     * @param trackId - The ID of the track.
     * @param frameIndex - The index of the frame.
     * @returns The closest detection.
     */
    getClosestDetectionAtFrame(trackId: number, frameIndex: number): TrackDetection {
        assert(frameIndex >= 0 && frameIndex < this.frameCount, 'Invalid frame index')
        const track = this.getTrackById(trackId)
        return getClosestDetectionAtFrame(track.detections, this.frameCount, frameIndex)
    }

    /**
     * Calculates the start and end frame indices for a given track based on its time percentages.
     * @param trackId - The ID of the track.
     * @returns An object containing startFrameIndex and endFrameIndex.
     */
    getTrackFrameRange(trackId: number): { startFrameIndex: number; endFrameIndex: number } {
        const track = this.getTrackById(trackId)
        const startFrameIndex = this.frameIndexForPercent(track.start_percent)
        const endFrameIndex = this.frameIndexForPercent(track.end_percent)
        return { startFrameIndex, endFrameIndex }
    }

    /**
     * Determines if a track is active at the specified frame index.
     * @param trackId - The ID of the track.
     * @param frameIndex - The index of the frame.
     * @returns True if the track is active at the given frame.
     */
    isTrackActiveAtFrame(trackId: number, frameIndex: number): boolean {
        const r = this.getTrackFrameRange(trackId)
        if (!r) return false
        return frameIndex >= r.startFrameIndex && frameIndex <= r.endFrameIndex
    }

    /**
     * Retrieves a track by its unique ID.
     * @param trackId - The ID of the track.
     * @returns The track object.
     */
    getTrackById(trackId: number): Track {
        const track = this.tracks.find(track => track.track_id === trackId)
        assert(track !== undefined, 'Track not found for id ${trackId}')
        return track!
    }

    /**
     * Converts a normalized time percentage (0.0 to 1.0) to a frame index.
     * @param percent - The time percentage.
     * @returns The corresponding frame index.
     */
    frameIndexForPercent(percent: number): number {
        return frameIndexForPercent(this.frameCount, percent)
    }
}
