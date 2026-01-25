import { processVideo } from '../../preprocess/preprocess'
import { QUALITY_HIGH } from 'mediabunny'
import { drawRotatedToCanvas } from './rotation'
import { PlayerState } from './state'
import { drawWatermark, getWatermarkAsset } from './watermark'
import { drawDetailedCrop, getSharedOffscreenCanvas } from './rendering'
import { DEFAULT_ZOOM_BASELINE } from './constants'

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

    // Always export in a wide format (16:9). We already rotate frames using `dominantOrientationDeg`.
    const outputWidth = 1920
    const outputHeight = 1080

    // Best-effort watermark; if it fails to load, we still export.
    const watermark = await getWatermarkAsset()

    const onFrame = async (current: { frame: VideoFrame; timestampSec: number; frameIndex: number }, ctx: Ctx2D) => {
        const frameIndex = Math.max(0, Math.min(player.frameCount - 1, current.frameIndex))
        const detection = player.getClosestDetectionAtFrame(trackId, frameIndex)

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
