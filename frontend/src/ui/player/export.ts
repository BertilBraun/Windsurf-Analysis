import { processVideo } from '../../preprocess/preprocess'
import { drawRotatedToCanvas } from './rotation'
import { PlayerState } from './state'
import { drawWatermark, getWatermarkAsset } from './watermark'
import { drawDetailedCrop, getSharedOffscreenCanvas, TimedBBox } from './rendering'

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

export function buildExportFilename(params: {
    sourceFileName: string | null
    localRelativePath: string | null | undefined
    trackId: number
    startSec: number
    endSec: number
}): string {
    const baseFromFile = params.sourceFileName ? stripExtension(basename(params.sourceFileName)) : ''
    const baseFromPath = params.localRelativePath ? stripExtension(basename(params.localRelativePath)) : ''
    const base = sanitizeFilenameBase(baseFromFile || baseFromPath)
    const start = params.startSec.toFixed(2)
    const end = params.endSec.toFixed(2)
    return `${base}_track_${params.trackId}_${start}-${end}.mp4`
}

export function downloadExport(blob: Blob, filename: string) {
    downloadBlob(blob, filename)
}

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

    const outputWidth = 1280
    const outputHeight = 720
    const bitrate = 8_000_000

    // Best-effort watermark; if it fails to load, we still export.
    const watermark = await getWatermarkAsset()

    let frameIndex = -1
    const onFrame = async (frame: VideoFrame, ctx: Ctx2D) => {
        frameIndex++
        const tSec = (frame.timestamp || 0) / 1_000_000

        if (tSec + 1e-6 < startSec) return false
        if (tSec >= endSec) return 'stop'

        const rotCanvas = getSharedOffscreenCanvas()
        const rotated = drawRotatedToCanvas(frame, rotCanvas, dominantOrientationDeg)

        const det0 = player.getDetectionAtFrame(trackId, frameIndex)
        const det: TimedBBox | null = det0 ? { time_percent: det0.time_percent, bbox: det0.bbox } : null
        drawDetailedCrop(ctx, outputWidth, outputHeight, rotCanvas, rotated.width, rotated.height, det)
        drawWatermark(ctx, outputWidth, outputHeight, watermark)
        return true
    }

    const outBuf = await processVideo({ file, onFrame, outputWidth, outputHeight, bitrate, onProgress })
    return new Blob([outBuf], { type: 'video/mp4' })
}
