import { MP4FrameSource } from './frame_source'
import { Mp4Encoder } from './mp4_encoder'

export type UploadQuality = 'original' | 'high' | 'medium' | 'minimum'

interface TargetSpec {
    longSide: number
    fps: number // 0 means "keep source fps"
}

function ensureEven(n: number): number {
    // Many codecs prefer even dimensions
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

// Very simple bitrate heuristic that scales with pixels and fps.
// Tuned so ~8 Mbps at 1080p30 for "high".
function estimateBitrate(width: number, height: number, fps: number, quality: UploadQuality): number {
    const mp = (width * height) / 1_000_000 // megapixels
    const fpsFactor = fps / 30
    const baseFor1080p30 = 4_000_000 // * mp ≈ 8 Mbps for 1080p30
    const qMul = quality === 'minimum' ? 0.45 : quality === 'medium' ? 0.7 : quality === 'high' ? 1.0 : 1.1 // 'original' - lean a bit higher
    const est = Math.round(baseFor1080p30 * mp * fpsFactor * qMul)
    // Keep within sensible bounds
    return Math.min(Math.max(est, 300_000), 20_000_000)
}

function chooseSpec(w: number, h: number, q: UploadQuality): TargetSpec {
    const longer = Math.max(w, h)
    if (q === 'original') return { longSide: longer, fps: 0 } // keep source fps
    if (q === 'high') return { longSide: longer >= 1920 ? 1920 : longer >= 1280 ? 1280 : 640, fps: 30 }
    if (q === 'medium') return { longSide: longer >= 1280 ? 1280 : 640, fps: 25 }
    return { longSide: 640, fps: 15 }
}

/**
 * Read a video file, downscale/reframe it by quality (resolution + fps),
 * and return an MP4 ArrayBuffer encoded with WebCodecs + mp4-muxer.
 *
 * Strategy:
 * - Demux+decode with MP4FrameSource
 * - Resize each kept frame onto a 2D canvas
 * - Pick frames using a Bresenham-style accumulator so we hit target FPS evenly
 * - Encode CFR with Mp4Encoder and return the resulting ArrayBuffer
 */
export async function preprocessVideo(
    file: File,
    quality: UploadQuality,
    onProgress?: (p: number) => void
): Promise<ArrayBuffer> {
    // Tiny reporter with throttling to avoid spamming the callback
    let lastReported = -1
    const report = (p: number) => {
        if (!onProgress) return
        const clamped = Math.max(0, Math.min(1, p))
        if (clamped === 1 || clamped === 0 || clamped - lastReported >= 0.005) {
            lastReported = clamped
            try {
                onProgress(clamped)
            } catch {}
        }
    }

    report(0)

    // 1) Read the file into memory
    const inputBuffer = await file.arrayBuffer()

    // 2) Set up the source (demux+decode)
    const src = new MP4FrameSource(inputBuffer)
    const info = await src.getTrackInfo()
    const srcW = info.width
    const srcH = info.height
    const srcFps = Math.max(1, info.approxFps)

    // 3) Decide target spec (resolution + fps)
    const spec = chooseSpec(srcW, srcH, quality)
    const { width: outW, height: outH } =
        quality === 'original'
            ? { width: ensureEven(srcW), height: ensureEven(srcH) }
            : fitToLongSide(srcW, srcH, spec.longSide)

    // Keep source FPS if spec.fps == 0; otherwise, never exceed source fps (we don’t synthesize extra frames here)
    const outFps = spec.fps === 0 ? srcFps : Math.min(spec.fps, srcFps)

    // 4) Choose a bitrate (codec-agnostic heuristic)
    const bitrate = estimateBitrate(outW, outH, outFps, quality)

    // 5) Set up encoder + canvas
    const enc = new Mp4Encoder({ width: outW, height: outH, fps: outFps, bitrate })
    const { canvas, ctx } = create2DCanvas(outW, outH)

    // total "units" for progress (prefer exact nbSamples; else fall back to duration * fps)
    const totalUnits = Math.max(1, info.nbSamples > 0 ? info.nbSamples : Math.round((info.durationSec || 0) * srcFps))
    let processedUnits = 0

    // 6) Frame selection using a Bresenham-style accumulator.
    //    ratio = outFps / srcFps. For each decoded frame: acc += ratio; if acc >= 1 → keep (emit), acc -= 1.
    const ratio = outFps / srcFps
    let acc = 0

    try {
        for await (const frame of src.frames()) {
            processedUnits++
            // Decode/demux dominates work → map to 0..0.95 of total
            // (Keeps UI snappy and avoids getting "stuck at 100%" while muxing)
            const decodePortion = 0.95
            report((processedUnits / totalUnits) * decodePortion)

            acc += ratio
            if (acc >= 1) {
                acc -= 1

                // Draw scaled
                // Clear is not strictly required if we always cover the full frame,
                // but it can help avoid artifacts when scaling down non-integer ratios.
                ctx.clearRect(0, 0, outW, outH)
                ctx.drawImage(frame, 0, 0, outW, outH)

                // Push to encoder at constant frame rate
                await enc.appendFrame(canvas)
            }

            // Always release decoded frames
            frame.close()
        }

        report(0.95)

        // 7) Finalize MP4 and return as ArrayBuffer
        const { blob } = await enc.finalize()

        report(1)

        return await blob.arrayBuffer()
    } finally {
        src.close()
        enc.destroy()
    }
}
