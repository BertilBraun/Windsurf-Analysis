import {
    ALL_FORMATS,
    BlobSource,
    BufferTarget,
    CanvasSource,
    Input,
    Mp4OutputFormat,
    Output,
    type Quality,
    QUALITY_HIGH,
    QUALITY_LOW,
    QUALITY_MEDIUM,
    VideoSampleSink,
} from 'mediabunny'

export type UploadQuality = 'original' | 'high' | 'medium' | 'minimum'

interface TargetSpec {
    longSide: number
    fps: number // 0 means "keep source fps"
}

function ensureEven(n: number): number {
    return n % 2 === 0 ? n : n - 1
}

function fitToLongSide(srcW: number, srcH: number, longSide: number): { width: number; height: number } {
    if (longSide <= 0) {
        return { width: ensureEven(srcW), height: ensureEven(srcH) }
    }
    const scale = longSide / Math.max(srcW, srcH)
    const width = ensureEven(Math.max(2, Math.round(srcW * scale)))
    const height = ensureEven(Math.max(2, Math.round(srcH * scale)))
    return { width, height }
}

function create2DCanvas(
    width: number,
    height: number
): {
    canvas: OffscreenCanvas | HTMLCanvasElement
    ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D
} {
    if (typeof OffscreenCanvas !== 'undefined') {
        const canvas = new OffscreenCanvas(width, height)
        const ctx = canvas.getContext('2d')
        if (!ctx) throw new Error('Failed to get 2D context')
        return { canvas, ctx }
    }
    if (typeof document !== 'undefined') {
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const ctx = canvas.getContext('2d')
        if (!ctx) throw new Error('Failed to get 2D context')
        return { canvas, ctx }
    }
    throw new Error('No canvas implementation available (OffscreenCanvas/DOM)')
}

function chooseSpec(w: number, h: number, q: UploadQuality): TargetSpec {
    const longer = Math.max(w, h)
    if (q === 'original') return { longSide: longer, fps: 0 } // keep source fps
    if (q === 'high') return { longSide: longer >= 1920 ? 1920 : longer >= 1280 ? 1280 : 640, fps: 30 }
    if (q === 'medium') return { longSide: longer >= 1280 ? 1280 : 640, fps: 25 }
    return { longSide: 640, fps: 15 }
}

function toExactArrayBuffer(buf: ArrayBuffer): ArrayBuffer {
    return buf.slice(0)
}

async function getApproxFps(videoTrack: Awaited<ReturnType<Input['getPrimaryVideoTrack']>>): Promise<number> {
    if (!videoTrack) return 30
    const stats = await videoTrack.computePacketStats(100)
    if (Number.isFinite(stats.averagePacketRate) && stats.averagePacketRate > 0) return stats.averagePacketRate
    return 30
}

export async function processVideo(params: {
    file: File
    inputStartSec?: number
    inputEndSec?: number
    outputWidth: number
    outputHeight: number
    /**
     * If omitted, uses the source video's approximate FPS.
     * The output is always constant frame rate (CFR).
     */
    outputFps?: number
    /**
     * Pass Mediabunny's `QUALITY_*` constants (recommended) or a numeric bitrate (bps).
     */
    videoBitrate?: number | Quality
    onProgress?: (p01: number) => void
    onFrame: (
        frame: VideoFrame,
        ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
        timestampSec: number,
        inputDurationSec: number | null
    ) => Promise<boolean>
}): Promise<ArrayBuffer> {
    const {
        file,
        inputStartSec,
        inputEndSec,
        outputWidth,
        outputHeight,
        outputFps,
        videoBitrate,
        onProgress,
        onFrame,
    } = params

    const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
    const output = new Output({ format: new Mp4OutputFormat(), target: new BufferTarget() })

    try {
        const videoTrack = await input.getPrimaryVideoTrack()
        if (!videoTrack) throw new Error('No video track found.')

        const inputDurationSec = await input.computeDuration().catch(() => null)
        const fps = Math.max(1e-6, outputFps ?? (await getApproxFps(videoTrack)))

        const { canvas, ctx } = create2DCanvas(outputWidth, outputHeight)
        const videoSource = new CanvasSource(canvas, { codec: 'avc', bitrate: videoBitrate ?? QUALITY_HIGH })
        output.addVideoTrack(videoSource, { frameRate: fps })

        await output.start()

        const expectedFrames =
            typeof inputStartSec === 'number' && typeof inputEndSec === 'number' && inputEndSec > inputStartSec
                ? Math.max(1, Math.ceil((inputEndSec - inputStartSec) * fps))
                : null

        let framesWritten = 0
        const sink = new VideoSampleSink(videoTrack)
        for await (const sample of sink.samples(inputStartSec, inputEndSec)) {
            const vf = sample.toVideoFrame()
            try {
                ctx.clearRect(0, 0, outputWidth, outputHeight)
                const keep = await onFrame(vf, ctx, sample.timestamp, inputDurationSec)
                if (!keep) continue

                await videoSource.add(framesWritten / fps, 1 / fps)
                framesWritten++

                if (onProgress && expectedFrames) onProgress(Math.min(0.95, (framesWritten / expectedFrames) * 0.95))
            } finally {
                try {
                    vf.close()
                } catch {}
                sample.close()
            }
        }

        if (onProgress) onProgress(0.95)
        await output.finalize()
        if (onProgress) onProgress(1)

        const buf = output.target.buffer
        if (!buf) throw new Error('Failed to retrieve output buffer.')
        return toExactArrayBuffer(buf)
    } finally {
        try {
            input.dispose()
        } catch {}
    }
}

export async function preprocessVideo(
    file: File,
    quality: UploadQuality,
    onProgress?: (p: number) => void,
    videoBitrate?: number | Quality
): Promise<ArrayBuffer> {
    // NOTE: Alternatively use https://mediabunny.dev/guide/converting-media-files

    const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
    let srcW = 1
    let srcH = 1
    let srcFps = 30
    try {
        const videoTrack = await input.getPrimaryVideoTrack()
        if (!videoTrack) throw new Error('No video track found.')
        srcW = videoTrack.displayWidth
        srcH = videoTrack.displayHeight
        srcFps = await getApproxFps(videoTrack)
    } finally {
        try {
            input.dispose()
        } catch {}
    }

    const spec = chooseSpec(srcW, srcH, quality)
    const { width: outW, height: outH } =
        quality === 'original'
            ? { width: ensureEven(srcW), height: ensureEven(srcH) }
            : fitToLongSide(srcW, srcH, spec.longSide)

    const outFps = spec.fps === 0 ? srcFps : Math.min(spec.fps, srcFps)
    const defaultVideoBitrate =
        quality === 'minimum'
            ? QUALITY_LOW
            : quality === 'medium'
            ? QUALITY_MEDIUM
            : quality === 'high'
            ? QUALITY_HIGH
            : QUALITY_HIGH

    const ratio = outFps / srcFps
    let acc = 0

    return processVideo({
        file,
        outputWidth: outW,
        outputHeight: outH,
        outputFps: outFps,
        videoBitrate: videoBitrate ?? defaultVideoBitrate,
        onProgress,
        onFrame: async (frame, ctx) => {
            ctx.drawImage(frame, 0, 0, ctx.canvas.width, ctx.canvas.height)
            acc += ratio
            if (acc >= 1) {
                acc -= 1
                return true
            }
            return false
        },
    })
}
