// mp4 demux + WebCodecs decode → async generator of VideoFrames (ESM/TS)
import { createFile, DataStream, MP4BoxBuffer } from 'https://cdn.jsdelivr.net/npm/mp4box@2.1.0/+esm'

type MP4BoxFile = ReturnType<typeof createFile>

export interface VideoTrackInfo {
    id: number
    timescale: number
    durationSec: number
    width: number
    height: number
    nbSamples: number
    codec: string
    approxFps: number
}

function normalizeCodecForWebCodecs(codec: string) {
    if (codec.startsWith('vp08')) return 'vp8'
    if (codec.startsWith('vp09')) return 'vp9'
    return codec
}

function getDescriptionFromStsd(mp4file: MP4BoxFile, vTrack: any): Uint8Array | undefined {
    const track = mp4file.getTrackById(vTrack.id)
    const entries = track.mdia.minf.stbl.stsd.entries
    for (const entry of entries) {
        const box = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C
        if (box) {
            const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN)
            box.write(stream)
            return new Uint8Array(stream.buffer, 8) // strip 8-byte MP4 box header
        }
    }
}

export class MP4FrameSource {
    private mp4: MP4BoxFile
    private buf: MP4BoxBuffer
    private ready: Promise<void>

    private vTrack!: any
    private timescale = 1
    private durationSec = 0
    private samplesRemaining = 0

    private decoder!: VideoDecoder
    private closed = false

    // simple async queue for VideoFrames
    private q: (VideoFrame | null)[] = []
    private waiters: ((v: VideoFrame | null) => void)[] = []

    constructor(arrayBuffer: ArrayBuffer) {
        this.buf = arrayBuffer as MP4BoxBuffer
        this.buf.fileStart = 0
        this.mp4 = createFile(true) // keep mdat
        this.ready = this.init()
    }

    /**
     * Returns parsed video track info (awaits internal initialization).
     */
    public async getTrackInfo(): Promise<VideoTrackInfo> {
        await this.ready
        const width = this.vTrack.video?.width || 1
        const height = this.vTrack.video?.height || 1
        const nb = this.vTrack.nb_samples || 0
        const approxFps = Math.max(1, Math.round(nb / (this.durationSec || 1)))
        return {
            id: this.vTrack.id,
            timescale: this.timescale,
            durationSec: this.durationSec,
            width,
            height,
            nbSamples: nb,
            codec: normalizeCodecForWebCodecs(this.vTrack.codec),
            approxFps,
        }
    }

    /**
     * Async generator yielding decoded VideoFrames in order.
     */
    public async *frames(): AsyncGenerator<VideoFrame, void, void> {
        await this.ready
        if (this.closed) return

        // Start extraction now that a consumer is ready
        this.mp4.start()

        while (true) {
            const next = await this.dequeue()
            if (next === null) break
            yield next
            next.close()
        }
    }

    /**
     * Close decoder and release resources.
     * Safe to call multiple times.
     */
    public close() {
        if (this.closed) return
        this.closed = true
        try {
            this.decoder.close()
        } catch {}
        // drain any pending waiters
        this.enqueue(null)
    }

    private enqueue(v: VideoFrame | null) {
        if (this.waiters.length) {
            const w = this.waiters.shift()!
            w(v)
        } else {
            this.q.push(v)
        }
    }

    private async dequeue(): Promise<VideoFrame | null> {
        if (this.q.length) {
            return this.q.shift()!
        }
        return new Promise<VideoFrame | null>(res => this.waiters.push(res))
    }

    private async init() {
        await new Promise<void>((resolve, reject) => {
            this.mp4.onError = reject
            this.mp4.onReady = (info: any) => {
                const v = info.videoTracks?.[0]
                if (!v) return reject(new Error('No video track found.'))
                this.vTrack = v
                this.timescale = v.timescale || 1
                this.durationSec = (v.duration || 0) / this.timescale
                this.samplesRemaining = v.nb_samples || 0
                resolve()
            }
            this.mp4.appendBuffer(this.buf)
            this.mp4.flush()
        })

        // Configure decoder
        const srcW = this.vTrack.video?.width || 1
        const srcH = this.vTrack.video?.height || 1
        const codec = normalizeCodecForWebCodecs(this.vTrack.codec)
        const description = getDescriptionFromStsd(this.mp4, this.vTrack)

        this.decoder = new VideoDecoder({
            output: (frame: VideoFrame) => {
                // push decoded frames in order
                this.enqueue(frame)
            },
            error: (e: any) => {
                console.error('[MP4FrameSource] Decoder error:', e)
                // surface error by ending the stream
                this.enqueue(null)
            },
        })

        this.decoder.configure({
            codec,
            codedWidth: srcW,
            codedHeight: srcH,
            description: description,
        })

        // Wire demux → decode
        this.mp4.setExtractionOptions(this.vTrack.id, null, { nbSamples: 1 })

        this.mp4.onSamples = (_trackId: number, _user: any, samples: any[]) => {
            for (const s of samples) {
                const tsUS = Math.round(((s.dts ?? s.cts) * 1e6) / this.timescale)
                const durUS = s.duration ? Math.round((s.duration * 1e6) / this.timescale) : 0

                const chunk = new EncodedVideoChunk({
                    type: s.is_sync ? 'key' : 'delta',
                    timestamp: tsUS,
                    duration: durUS,
                    data: s.data instanceof Uint8Array ? s.data : new Uint8Array(s.data),
                })
                try {
                    this.decoder.decode(chunk)
                } catch (e) {
                    console.error('[MP4FrameSource] decode() failed:', e)
                }
            }
            this.samplesRemaining -= samples.length
            if (this.samplesRemaining <= 0) {
                // End of stream → flush and mark done
                this.decoder.flush().finally(() => {
                    this.enqueue(null) // sentinel: no more frames
                })
            }
        }

        this.mp4.onError = () => console.error('[MP4FrameSource] mp4.onError')
    }
}
