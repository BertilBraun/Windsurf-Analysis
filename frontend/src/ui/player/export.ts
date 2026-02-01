/**
 * Utilities for exporting processed video tracks from the player.
 * Handles filename generation, browser downloads, and the core MP4 export logic.
 */

import { processVideo } from '../../preprocess/preprocess'
import { QUALITY_HIGH } from 'mediabunny'
import { drawRotatedToCanvas } from './rotation'
import { PlayerState } from './state'
import { drawWatermark, getWatermarkAsset } from './watermark'
import { drawDetailedCrop, getSharedOffscreenCanvas } from './rendering'
import { DEFAULT_ZOOM_BASELINE } from './constants'
import { assert } from '../utils/assert'

type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D

function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.style.display = 'none'
    document.body.appendChild(a)
    a.click()
    a.remove()
    // Give the download a moment to start before revoking.
    setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function sanitizeFilenameBase(name: string): string {
    // Basic Windows/macOS-friendly sanitization.
    const cleaned = name.replace(/[<>:"/\\|?*\x00-\x1F]+/g, '_').trim()
    return cleaned.length > 0 ? cleaned : 'export'
}

function basename(path: string): string {
    const parts = path.split(/[\\/]+/).filter(Boolean)
    return parts.length ? parts[parts.length - 1] : path
}

function stripExtension(name: string): string {
    return name.replace(/\.[^./\\]+$/, '')
}

/**
 * Constructs a standardized filename for a video export based on source metadata and track timing.
 *
 * @param params - Metadata and timing for the export.
 * @returns A sanitized filename string.
 */
export function buildExportFilename(params: {
    sourceFileName: string
    localRelativePath: string | null | undefined
    trackId: number
    startSec: number
    endSec: number
}): string {
    const base = sanitizeFilenameBase(stripExtension(basename(params.sourceFileName)))
    const start = params.startSec.toFixed(2)
    const end = params.endSec.toFixed(2)
    return `${base}_track_${params.trackId}_${start}-${end}.mp4`
}

/**
 * Triggers a browser download for the provided Blob.
 *
 * @param blob - The data to download.
 * @param filename - The name of the file to be saved.
 */
export function downloadExport(blob: Blob, filename: string) {
    downloadBlob(blob, filename)
}

type ShareCapableNavigator = Navigator & {
    share?: (data: ShareData) => Promise<void>
    canShare?: (data: ShareData) => boolean
}

export function canShareExport(blob: Blob, filename: string): boolean {
    if (typeof navigator === 'undefined') return false
    const nav = navigator as ShareCapableNavigator
    if (typeof nav.share !== 'function') return false
    if (typeof File === 'undefined') return false

    const file = new File([blob], filename, { type: blob.type || 'video/mp4' })
    if (typeof nav.canShare !== 'function') return true
    return nav.canShare({ files: [file] })
}

export async function shareExport(params: { blob: Blob; filename: string; text?: string; title?: string }) {
    const { blob, filename, text, title } = params
    const nav = navigator as ShareCapableNavigator
    if (typeof nav.share !== 'function') throw new Error('Share not supported')
    if (typeof File === 'undefined') throw new Error('Share not supported')
    const file = new File([blob], filename, { type: blob.type || 'video/mp4' })
    await nav.share({ files: [file], text, title })
}

/**
 * Processes a video segment to export a specific track as an MP4.
 * Applies rotation, dynamic cropping based on track detections, and watermarking.
 *
 * @param params - Configuration for the export process.
 * @param params.file - The source video file.
 * @param params.player - Player state containing track detections.
 * @param params.dominantOrientationDeg - Rotation to apply to source frames.
 * @param params.trackId - ID of the track to follow.
 * @param params.startSec - Start time in seconds.
 * @param params.endSec - End time in seconds.
 * @param params.onProgress - Optional progress callback (0-1).
 * @returns A Promise resolving to the exported MP4 Blob.
 */
export async function exportTrackMp4(params: {
    file: File
    player: PlayerState
    dominantOrientationDeg: number
    trackId: number
    startSec: number
    endSec: number
    onProgress?: (p01: number) => void
}): Promise<Blob> {
    const { file, player, dominantOrientationDeg, trackId, startSec, endSec, onProgress } = params

    // Always export in a wide format (16:9). We already rotate frames using `dominantOrientationDeg`.
    const outputWidth = 1920
    const outputHeight = 1080

    // Best-effort watermark; if it fails to load, we still export.
    const watermark = await getWatermarkAsset()

    const onFrame = async (current: { frame: VideoFrame; timestampSec: number; frameIndex: number }, ctx: Ctx2D) => {
        assert(0 <= current.frameIndex && current.frameIndex < player.frameCount)
        const detection = player.getClosestDetectionAtFrame(trackId, current.frameIndex)

        const rotCanvas = getSharedOffscreenCanvas()
        const rotatedFrame = drawRotatedToCanvas(current.frame, rotCanvas, dominantOrientationDeg)
        drawDetailedCrop(
            ctx,
            outputWidth,
            outputHeight,
            rotCanvas,
            rotatedFrame.width,
            rotatedFrame.height,
            detection,
            DEFAULT_ZOOM_BASELINE
        )
        drawWatermark(ctx, outputWidth, outputHeight, watermark)
        return true
    }

    const outBuf = await processVideo({
        file,
        onFrame,
        inputStartSec: startSec,
        inputEndSec: endSec,
        outputWidth,
        outputHeight,
        videoBitrate: QUALITY_HIGH,
        onProgress,
    })
    return new Blob([outBuf], { type: 'video/mp4' })
}
