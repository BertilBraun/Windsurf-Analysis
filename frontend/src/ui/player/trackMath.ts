import { TrackDetection } from '../types'
import { assert } from '../utils/assert'

export function frameIndexForPercent(frameCount: number, percent: number): number {
    assert(frameCount > 0, 'Invalid frame count')
    assert(percent >= 0 && percent <= 1, 'Invalid percent')
    const frameIndex = Math.round(percent * frameCount)
    return Math.max(0, Math.min(frameCount - 1, frameIndex))
}

export function getClosestDetectionAtFrame(
    detections: TrackDetection[],
    frameCount: number,
    frameIndex: number
): TrackDetection {
    assert(frameCount > 0, 'Invalid frame count')
    assert(frameIndex >= 0 && frameIndex < frameCount, 'Invalid frame index')
    assert(detections.length > 0, 'No detections available')

    let closest = detections[0]!
    let closestDistance = Math.abs(frameIndex - frameIndexForPercent(frameCount, closest.time_percent))
    for (const detection of detections) {
        const distance = Math.abs(frameIndex - frameIndexForPercent(frameCount, detection.time_percent))
        if (distance < closestDistance) {
            closest = detection
            closestDistance = distance
        }
    }
    return closest
}

