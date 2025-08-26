import * as MP4Box from 'mp4box'
import { UploadQuality } from '../types'

interface TargetSpec {
    longSide: number
    fps: number // 0 => keep source FPS
    bitrate: number // in bps; 0 => derive from source
}

interface TrackInfo {
    id: number
    width: number
    height: number
    timescale: number // track timebase
    duration: number // in track timescale
    codec: string // e.g., "avc1.640032"
    nb_samples: number // number of samples in track
    bitrate?: number
}

/* ----------------------------- logging helper ----------------------------- */
const log = (...args: any[]) => console.log('[preprocessVideo]', ...args)

/* -------------------- choose resolution / fps / bitrate ------------------- */
function chooseSpec(width: number, height: number, quality: UploadQuality): TargetSpec {
    const longer = Math.max(width, height)

    const ladder = {
        high: 4_000_000, // ~4 Mbps
        medium: 2_000_000, // ~2 Mbps
        minimum: 800_000, // ~0.8 Mbps
    }

    if (quality === 'original') {
        return { longSide: longer, fps: 0, bitrate: 0 }
    }
    if (quality === 'high') {
        if (longer >= 1920) return { longSide: 1920, fps: 30, bitrate: ladder.high }
        if (longer >= 1280) return { longSide: 1280, fps: 30, bitrate: ladder.high }
        return { longSide: 640, fps: 30, bitrate: ladder.high }
    }
    if (quality === 'medium') {
        return { longSide: longer >= 1280 ? 1280 : 640, fps: 25, bitrate: ladder.medium }
    }
    // minimum
    return { longSide: 640, fps: 15, bitrate: ladder.minimum }
}

/* --------- conservative AVC level (hex) selection by frame area ----------- */
function chooseAvcLevelHex(width: number, height: number): string {
    const area = width * height
    // L3.0 (0x1E) ≈ 720x576, L3.1 (0x1F) ≈ 1280x720, L4.0 (0x28) ≈ 1920x1080
    if (area <= 414_720) return '1E' // 3.0
    if (area <= 921_600) return '1F' // 3.1
    if (area <= 2_073_600) return '28' // 4.0
    return '29' // 4.1 fallback
}

/* --------------------------- canvas draw helper --------------------------- */
function drawScaled(ctx: CanvasRenderingContext2D, frame: VideoFrame, outW: number, outH: number) {
    // simple fit (no letterbox): preserve AR by sizing the canvas to the exact targetW/targetH we computed
    ctx.drawImage(frame as any, 0, 0, outW, outH)
}

/* ----------------------------- encoder factory ---------------------------- */
function createEncoder(outW: number, outH: number, fps: number, bitrate: number) {
    let decoderConfig: VideoDecoderConfig | undefined
    const chunks: EncodedVideoChunk[] = []

    const encoder = new VideoEncoder({
        output: (chunk, meta) => {
            if (meta?.decoderConfig && !decoderConfig) decoderConfig = meta.decoderConfig
            chunks.push(chunk)
        },
        error: e => log('Encoder error:', e),
    })

    const codec = `avc1.42E0${chooseAvcLevelHex(outW, outH)}` // H.264 Baseline + level
    const cfg: VideoEncoderConfig = {
        codec,
        width: outW,
        height: outH,
        framerate: fps > 0 ? fps : 30, // fallback
        bitrate: Math.max(100_000, bitrate || 2_000_000), // safety floor
        hardwareAcceleration: 'prefer-hardware',
    }

    log('Encoder.configure', cfg)
    encoder.configure(cfg)

    return {
        encoder,
        chunks,
        getDecoderConfig: () => decoderConfig,
    }
}

/* ------------------------------ muxer factory ----------------------------- */
function createMp4Muxer(outW: number, outH: number, avcC: ArrayBufferLike) {
    const mux = (MP4Box as any).createFile()
    const timescale = 1_000_000 // microseconds
    const segs: ArrayBuffer[] = []

    mux.onSegment = (_id: number, _user: any, buffer: ArrayBuffer) => {
        segs.push(buffer)
    }

    const trackId = mux.addTrack({
        timescale,
        width: outW,
        height: outH,
        hdlr: 'vide',
        avcDecoderConfigRecord: avcC,
    })

    mux.setSegmentOptions(trackId, null, { nbSamples: 1_000_000_000 })
    mux.initializeSegmentation()
    mux.start()

    function addChunk(ch: EncodedVideoChunk, fps: number) {
        const u8 = new Uint8Array(ch.byteLength)
        ch.copyTo(u8)
        // MP4Box expects a sample-like object; dts/cts in "timescale" units
        const sample = {
            duration: ch.duration ?? Math.round(1e6 / fps),
            dts: ch.timestamp,
            cts: 0,
            is_sync: ch.type === 'key',
            size: u8.byteLength,
        }
        // hint property MP4Box uses internally (ok to set 0)
        ;(u8.buffer as any).fileStart = 0
        mux.addSample(trackId, u8.buffer, sample)
    }

    function finalize(): ArrayBuffer {
        mux.flush()
        let total = 0
        for (const s of segs) total += s.byteLength
        const out = new Uint8Array(total)
        let off = 0
        for (const s of segs) {
            out.set(new Uint8Array(s), off)
            off += s.byteLength
        }
        return out.buffer
    }

    return { addChunk, finalize }
}

/* -------------------------- demux+decode (fast) --------------------------- */
/**
 * Stream the file to MP4Box in chunks so `onSamples` fires while data arrives.
 * Push video samples into WebCodecs VideoDecoder as EncodedVideoChunk.
 */
async function decodeAllFramesFast(
    fileBuf: ArrayBuffer,
    onReadyTrack: (ti: TrackInfo) => void,
    onDecodedFrame: (frame: VideoFrame) => void
): Promise<void> {
    return new Promise<void>((resolve, reject) => {
        const mp4 = (MP4Box as any).createFile()
        const CHUNK = 1 << 20 // 1 MiB
        let offset = 0
        let videoTrackId: number | null = null
        let trackInfo: TrackInfo | null = null
        let decoder: VideoDecoder | null = null

        mp4.onError = (e: any) => reject(e)

        mp4.onReady = (info: any) => {
            log('MP4Box.onReady', info)
            const vt = info.tracks.find((t: any) => !!t.video)
            if (!vt) {
                reject(new Error('No video track'))
                return
            }
            videoTrackId = vt.id
            trackInfo = {
                id: vt.id,
                width: vt.video.width,
                height: vt.video.height,
                timescale: vt.timescale,
                duration: vt.duration,
                codec: vt.codec,
                nb_samples: vt.nb_samples,
                bitrate: vt.bitrate,
            }
            onReadyTrack(trackInfo)

            // Configure decoder
            decoder = new VideoDecoder({
                output: frame => onDecodedFrame(frame),
                error: e => reject(e),
            })
            decoder.configure({ codec: vt.codec })

            // Start extraction NOW so further appended data yields samples
            mp4.setExtractionOptions(videoTrackId, null, { nbSamples: vt.nb_samples || 0 })
            mp4.start()
        }

        mp4.onSamples = (id: number, _user: any, samples: any[]) => {
            if (id !== videoTrackId || !decoder) return
            log('onSamples batch', samples.length)
            for (const s of samples) {
                const tsUs = Math.round((s.dts / trackInfo!.timescale) * 1_000_000)
                const durUs = s.duration ? Math.round((s.duration / trackInfo!.timescale) * 1_000_000) : undefined
                const chunk = new EncodedVideoChunk({
                    type: s.is_sync ? 'key' : 'delta',
                    timestamp: tsUs,
                    duration: durUs,
                    data: new Uint8Array(s.data),
                })
                decoder.decode(chunk)
            }
        }

        // Stream the file into MP4Box
        const u8 = new Uint8Array(fileBuf)
        const next = () => {
            if (offset >= u8.byteLength) {
                log('All data appended, flushing MP4Box…')
                mp4.flush()
                // wait for decoder to drain
                const dec = decoder!
                dec.flush()
                    .then(() => {
                        dec.close()
                        resolve()
                    })
                    .catch(reject)
                return
            }
            const end = Math.min(offset + CHUNK, u8.byteLength)
            const slice = u8.subarray(offset, end)
            // MP4Box needs a buffer that carries a fileStart property
            const buf = slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength) as ArrayBuffer & {
                fileStart?: number
            }
            ;(buf as any).fileStart = offset
            mp4.appendBuffer(buf)
            offset = end
            // schedule next chunk (yield to UI thread)
            setTimeout(next, 0)
        }

        next()
    })
}

/* ------------------------------- main API -------------------------------- */
export async function preprocessVideo(file: File, quality: UploadQuality): Promise<ArrayBuffer> {
    if (!('VideoDecoder' in window) || !('VideoEncoder' in window)) {
        throw new Error('WebCodecs not supported in this browser.')
    }

    const fileBuf = await file.arrayBuffer()
    log('Begin preprocess', { name: file.name, bytes: fileBuf.byteLength, quality })

    let srcTrack: TrackInfo | null = null

    // these will be set after we know source dimensions & target spec
    let targetW = 0,
        targetH = 0,
        targetFps = 0,
        targetBitrate = 0
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d', { alpha: false })!

    // encoder/muxer will be created lazily on first decoded frame,
    // once we have the target sizes and have configured the encoder.
    let encoderHandle: ReturnType<typeof createEncoder> | null = null
    let muxer: ReturnType<typeof createMp4Muxer> | null = null

    // frame pacing for exact FPS output (drop/dup as needed)
    let haveFpsPlan = false
    let nextTsUs = 0 // synthetic DTS for output stream in microseconds
    let frameIntervalUs = 0

    const onReadyTrack = (ti: TrackInfo) => {
        srcTrack = ti
        const spec = chooseSpec(ti.width, ti.height, quality)

        // compute target W/H preserving AR (no upscaling beyond chosen longSide)
        const aspect = ti.width / ti.height
        targetW = ti.width >= ti.height ? spec.longSide : Math.round(spec.longSide * aspect)
        targetH = ti.height > ti.width ? spec.longSide : Math.round(spec.longSide / aspect)

        // enforce multiples of 2 (H.264 requirement)
        if (targetW % 2) targetW += 1
        if (targetH % 2) targetH += 1

        // fps and bitrate
        const srcFpsApprox =
            ti.nb_samples && ti.duration ? Math.max(1, Math.round((ti.nb_samples * ti.timescale) / ti.duration)) : 30

        targetFps = spec.fps || srcFpsApprox
        targetBitrate = spec.bitrate || ti.bitrate || 2_000_000

        canvas.width = targetW
        canvas.height = targetH

        log('Source track', ti)
        log('Target spec', { targetW, targetH, targetFps, targetBitrate })
    }

    const onDecodedFrame = (frame: VideoFrame) => {
        // lazily create encoder/muxer when first frame arrives
        if (!encoderHandle) {
            encoderHandle = createEncoder(targetW, targetH, targetFps, targetBitrate)
            const decCfg = encoderHandle.getDecoderConfig() // may be undefined until first output
            // We'll initialize muxer once we actually get decoderConfig in encoder.output.
            haveFpsPlan = true
            frameIntervalUs = Math.round(1e6 / targetFps)
            nextTsUs = 0
        }

        // scale into canvas
        drawScaled(ctx, frame, targetW, targetH)

        // synthesize constant-FPS timestamps (don’t rely on source dts)
        const outTs = nextTsUs
        nextTsUs += frameIntervalUs

        // create a VideoFrame from the canvas and push to encoder
        const outFrame = new VideoFrame(canvas, { timestamp: outTs })
        const makeKey = outTs === 0 || (outTs / 1e6) % 2 === 0 // ~2s GOP
        encoderHandle!.encoder.encode(outFrame, { keyFrame: makeKey })
        outFrame.close()

        // if encoder just emitted its first chunk and gave us decoderConfig, start muxer
        const dc = encoderHandle!.getDecoderConfig()
        if (dc && !muxer) {
            // normalize description → Uint8Array/ArrayBuffer
            const desc = dc.description!
            let avcC: ArrayBufferLike
            if (desc instanceof ArrayBuffer) avcC = desc
            else if (ArrayBuffer.isView(desc as ArrayBufferView)) {
                const v = desc as ArrayBufferView
                avcC = v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength)
            } else {
                throw new Error('DecoderConfig.description missing')
            }
            muxer = createMp4Muxer(targetW, targetH, avcC)
            log('Muxer initialized')
        }

        frame.close()
    }

    // demux+decode everything (fast path, chunked append)
    await decodeAllFramesFast(fileBuf, onReadyTrack, onDecodedFrame)

    // drain encoder and build mp4
    if (!encoderHandle) {
        throw new Error('No frames decoded.')
    }

    log('Flushing encoder…')
    await encoderHandle.encoder.flush()
    encoderHandle.encoder.close()

    // feed all chunks to muxer
    if (!muxer) {
        // should not happen; guard anyway
        const dc = encoderHandle.getDecoderConfig()
        if (!dc) throw new Error('No decoderConfig for muxer')
        const desc = dc.description! as ArrayBuffer
        muxer = createMp4Muxer(targetW, targetH, desc)
    }

    log(`Muxing ${encoderHandle.chunks.length} chunks…`)
    for (const ch of encoderHandle.chunks) {
        muxer.addChunk(ch, targetFps)
    }

    const outBuf = muxer.finalize()
    log('Done. Output bytes:', outBuf.byteLength)
    return outBuf
}
