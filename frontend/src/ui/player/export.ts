import { processVideo } from '../../preprocess/preprocess'
import { QUALITY_HIGH } from 'mediabunny'
import { drawRotatedToCanvas } from './rotation'
import { PlayerState } from './state'
import { drawWatermark, getWatermarkAsset } from './watermark'
import { drawDetailedCrop, getSharedOffscreenCanvas } from './rendering'

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

    const outputWidth = 1920
    const outputHeight = 1920

    // Best-effort watermark; if it fails to load, we still export.
    const watermark = await getWatermarkAsset()

    const onFrame = async (frame: VideoFrame, ctx: Ctx2D, tSec: number, inputDurationSec: number | null) => {
        const dur = inputDurationSec
        const frameIndex =
            dur && dur > 0 ? player.frameIndexForPercent(Math.max(0, Math.min(1, tSec / dur))) : player.frameCount - 1

        const detection = player.getClosestDetectionAtFrame(trackId, frameIndex)

        const rotCanvas = getSharedOffscreenCanvas()
        const rotated = drawRotatedToCanvas(frame, rotCanvas, dominantOrientationDeg)
        drawDetailedCrop(ctx, outputWidth, outputHeight, rotCanvas, rotated.width, rotated.height, detection, 1)
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
