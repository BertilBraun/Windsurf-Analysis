import { Muxer, ArrayBufferTarget } from 'mp4-muxer'
import { selectSupportedCodec, SelectedCodecInfo } from './codec'

export interface Mp4EncoderOptions {
    /** Output width in pixels (must match your render surface). */
    width: number
    /** Output height in pixels (must match your render surface). */
    height: number
    /** Constant output frame rate (e.g., 30). */
    fps: number
    /** Target bitrate in bits per second (e.g., 8_000_000). */
    bitrate?: number
}

/** Thin wrapper so callers can stream out data if they prefer. */
export interface FinalizeResult {
    blob: Blob
    /** Size in bytes of the MP4 in memory (for logging/metrics). */
    byteLength: number
}

/**
 * Offline, faster-than-real-time encoder for constant frame rate MP4.
 * You push frames explicitly; timestamps are synthesized for exact CFR.
 */
export class Mp4Encoder {
    private width: number
    private height: number
    private fps: number
    private bitrate: number

    private ready: Promise<void>
    private encoder!: VideoEncoder
    private muxer!: Muxer<ArrayBufferTarget>
    private target!: ArrayBufferTarget
    private frameIndex = 0
    private selected!: SelectedCodecInfo

    constructor(opts: Mp4EncoderOptions) {
        const { width, height, fps, bitrate = 8_000_000 } = opts

        this.width = width
        this.height = height
        this.fps = fps
        this.bitrate = bitrate

        this.ready = this._setup()
    }

    private async _setup() {
        // 1) Decide codec
        this.selected = await selectSupportedCodec({
            width: this.width,
            height: this.height,
            fps: this.fps,
            bitrate: this.bitrate,
        })

        // 2) Set up mp4-muxer
        this.target = new ArrayBufferTarget()
        this.muxer = new Muxer({
            target: this.target,
            fastStart: 'in-memory',
            video: {
                codec: this.selected.muxerKind,
                width: this.width,
                height: this.height,
                frameRate: this.fps,
            },
        })

        // 3) Configure WebCodecs encoder
        this.encoder = new VideoEncoder({
            output: (chunk, meta) => this.muxer.addVideoChunk(chunk, meta),
            error: e => console.error('[VideoEncoder] error:', e),
        })

        this.encoder.configure({
            ...this.selected.config,
            // Ensure our desired bitrate/framerate are present (some UAs normalize values)
            width: this.width,
            height: this.height,
            bitrate: this.bitrate,
            framerate: this.fps,
            bitrateMode: this.selected.config.bitrateMode ?? 'variable',
            hardwareAcceleration: 'prefer-hardware',
            latencyMode: 'realtime', // low-latency pipeline; fine for offline too
        })
    }

    /** Returns the final codec picked by auto-selection (call after construction). */
    async getSelectedCodec(): Promise<string> {
        await this.ready
        return this.selected.codec
    }

    /**
     * Append one or more frames at constant frame rate.
     * If you want to hold the same visual for N frames, set repeatFrames > 1.
     *
     * @param source Render surface for this frame (OffscreenCanvas, HTMLCanvas, ImageBitmap, VideoFrame…)
     */
    async appendFrame(source: CanvasImageSource): Promise<void> {
        await this.ready

        while (this.encoder.encodeQueueSize > 10) {
            await new Promise(resolve => setTimeout(resolve, 10))
        }

        const usPerFrame = Math.round(1_000_000 / this.fps)

        const timestamp = this.frameIndex * usPerFrame
        const vf = new VideoFrame(source, { timestamp })

        this.encoder.encode(vf, { keyFrame: this.frameIndex % this.fps === 0 })
        if (vf !== source) vf.close()
        this.frameIndex++
    }

    /** Flush encoder & finalize MP4. */
    async finalize(): Promise<FinalizeResult> {
        await this.ready
        await this.encoder.flush()
        this.muxer.finalize()

        const buffer = this.target.buffer // ArrayBuffer
        const blob = new Blob([buffer], { type: 'video/mp4' })
        return { blob, byteLength: buffer.byteLength }
    }

    /** Free resources early if you abort a render. */
    destroy(): void {
        try {
            this.encoder?.close()
        } catch {}
    }
}
