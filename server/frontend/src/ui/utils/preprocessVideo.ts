// src/ui/utils/preprocessVideo.ts
/* eslint-disable no-console */

// mp4box is UMD; add a shim typing in your project, e.g.:
// declare module "mp4box" { export function createFile(): any; }
import * as MP4Box from 'mp4box'

export type UploadQuality = 'original' | 'high' | 'medium' | 'minimum'

interface TargetSpec {
    longSide: number
    fps: number // 0 => keep source fps
    bitrate: number // bps; 0 => derive from source
}

interface TrackInfo {
    id: number
    width: number
    height: number
    timescale: number
    duration: number
    codec: string // e.g. "avc1.640032" or "avc3.640032"
    nb_samples: number
    bitrate?: number
}

const log = (...args: any[]) => console.log('[preprocessVideo]', ...args)

/* -------------------- choose resolution / fps / bitrate ------------------- */
function chooseSpec(width: number, height: number, quality: UploadQuality): TargetSpec {
    const longer = Math.max(width, height)

    const ladder = {
        high: 4_000_000,
        medium: 2_000_000,
        minimum: 800_000,
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
    if (area <= 414_720) return '1E' // L3.0
    if (area <= 921_600) return '1F' // L3.1
    if (area <= 2_073_600) return '28' // L4.0
    return '29' // L4.1
}

/* --------------------------- canvas draw helper --------------------------- */
function drawScaled(ctx: CanvasRenderingContext2D, frame: VideoFrame, outW: number, outH: number) {
    ctx.drawImage(frame as any, 0, 0, outW, outH)
}

/* ------------------------------ NAL helpers ------------------------------- */
const NALU_TYPE_SPS = 7
const NALU_TYPE_PPS = 8

/** Try to parse a key sample (length‑prefixed AVCC or Annex B) and build an avcC buffer */
function buildAvcCFromKeySample(sampleData: ArrayBuffer): ArrayBuffer | null {
    const u8 = new Uint8Array(sampleData)

    // Heuristics: if we see 0x00 00 00 01 start codes, treat as Annex B; otherwise assume 4‑byte length‑prefixed (AVCC)
    const isAnnexB =
        u8.length >= 4 && u8[0] === 0x00 && u8[1] === 0x00 && ((u8[2] === 0x00 && u8[3] === 0x01) || u8[2] === 0x01)

    const spsList: Uint8Array[] = []
    const ppsList: Uint8Array[] = []

    if (isAnnexB) {
        // Split by start codes
        let i = 0
        function nextStart(pos: number) {
            for (let j = pos + 3; j < u8.length; j++) {
                if (
                    u8[j - 3] === 0x00 &&
                    u8[j - 2] === 0x00 &&
                    ((u8[j - 1] === 0x00 && u8[j] === 0x01) || u8[j - 1] === 0x01)
                )
                    return j - (u8[j - 1] === 0x01 ? 3 : 4)
            }
            return u8.length
        }
        while (i < u8.length) {
            // find start prefix
            if (
                !(
                    u8[i] === 0x00 &&
                    u8[i + 1] === 0x00 &&
                    ((u8[i + 2] === 0x00 && u8[i + 3] === 0x01) || u8[i + 2] === 0x01)
                )
            ) {
                i++
                continue
            }
            const start = u8[i + 2] === 0x01 ? i + 3 : i + 4
            const next = nextStart(start)
            const nal = u8.subarray(start, next)
            const nalType = nal[0] & 0x1f
            if (nalType === NALU_TYPE_SPS) spsList.push(nal)
            else if (nalType === NALU_TYPE_PPS) ppsList.push(nal)
            i = next
        }
    } else {
        // AVCC: 4‑byte BE lengths
        let off = 0
        while (off + 4 <= u8.length) {
            const len = (u8[off] << 24) | (u8[off + 1] << 16) | (u8[off + 2] << 8) | u8[off + 3]
            off += 4
            if (len <= 0 || off + len > u8.length) break
            const nal = u8.subarray(off, off + len)
            off += len
            const nalType = nal[0] & 0x1f
            if (nalType === NALU_TYPE_SPS) spsList.push(nal)
            else if (nalType === NALU_TYPE_PPS) ppsList.push(nal)
        }
    }

    if (!spsList.length || !ppsList.length) {
        log('buildAvcCFromKeySample: SPS/PPS not found in keyframe')
        return null
    }

    // Construct AVCDecoderConfigurationRecord (ISO/IEC 14496‑15)
    const sps = spsList[0]
    const profile_idc = sps[1]
    const constraint_set_flags = sps[2]
    const level_idc = sps[3]

    // lengthSizeMinusOne = 3 (4‑byte lengths)
    let size = 7 + 2 + spsList.reduce((a, n) => a + 2 + n.length, 0) + 1 + ppsList.reduce((a, n) => a + 2 + n.length, 0)
    const avcC = new Uint8Array(size)
    let p = 0
    avcC[p++] = 1 // configurationVersion
    avcC[p++] = profile_idc // AVCProfileIndication
    avcC[p++] = constraint_set_flags // profile_compatibility
    avcC[p++] = level_idc // AVCLevelIndication
    avcC[p++] = 0xff // lengthSizeMinusOne (3) | reserved
    // SPS
    avcC[p++] = 0xe0 | (spsList.length & 0x1f) // numOfSPS
    for (const n of spsList) {
        avcC[p++] = (n.length >>> 8) & 0xff
        avcC[p++] = n.length & 0xff
        avcC.set(n, p)
        p += n.length
    }
    // PPS
    avcC[p++] = ppsList.length & 0xff // numOfPPS
    for (const n of ppsList) {
        avcC[p++] = (n.length >>> 8) & 0xff
        avcC[p++] = n.length & 0xff
        avcC.set(n, p)
        p += n.length
    }

    log('Synthesized avcC from keyframe SPS/PPS (profile', profile_idc, 'level', level_idc, ')')
    return avcC.buffer
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
        framerate: fps > 0 ? fps : 30,
        bitrate: Math.max(100_000, bitrate || 2_000_000),
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
        const sample = {
            duration: ch.duration ?? Math.round(1e6 / fps),
            dts: ch.timestamp,
            cts: 0,
            is_sync: ch.type === 'key',
            size: u8.byteLength,
        }
        ;(u8.buffer as any).fileStart = 0 // hint required by MP4Box
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
async function decodeAllFramesFast(
    fileBuf: ArrayBuffer,
    onReadyTrack: (ti: TrackInfo, avcC: ArrayBuffer | null) => void,
    onDecodedFrame: (frame: VideoFrame) => void
): Promise<void> {
    return new Promise<void>((resolve, reject) => {
        const mp4 = (MP4Box as any).createFile()
        const CHUNK = 1 << 20 // 1 MiB
        let offset = 0

        let videoTrackId: number | null = null
        let trackInfo: TrackInfo | null = null
        let inputAvcC: ArrayBuffer | null = null

        // Decoder is created once we have avcC (either from metadata or synthesized)
        let decoder: VideoDecoder | null = null
        let configured = false
        let seenKey = false

        mp4.onError = (e: any) => reject(e)

        mp4.onReady = (info: any) => {
            log('MP4Box.onReady', info)
            const vt = info.tracks.find((t: any) => !!t.video)
            if (!vt) return reject(new Error('No video track'))

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

            inputAvcC = extractInputAvcC(mp4, vt)
            onReadyTrack(trackInfo, inputAvcC)

            // start extraction immediately; align on RAP so the first sample we see can be a keyframe
            mp4.setExtractionOptions(videoTrackId, null, {
                nbSamples: vt.nb_samples || 0,
                rap: true,
                rapAlignment: true, // NOTE: correct spelling
            })
            mp4.start()

            if (inputAvcC) {
                // configure immediately using avcC from metadata
                decoder = new VideoDecoder({
                    output: frame => onDecodedFrame(frame),
                    error: e => reject(e),
                })
                const decCfg: VideoDecoderConfig = { codec: vt.codec, description: inputAvcC }
                log('Decoder.configure (with avcC)', { codec: decCfg.codec, hasDescription: !!decCfg.description })
                decoder.configure(decCfg)
                configured = true
            } else {
                log('No avcC in metadata; will synthesize from first keyframe')
            }
        }

        mp4.onSamples = (id: number, _user: any, samples: any[]) => {
            if (id !== videoTrackId) return
            log('onSamples batch', samples.length)

            for (const s of samples) {
                // Wait until we have a decoder configured: either already from metadata or synthesized from first key
                if (!configured) {
                    if (!s.is_sync) continue // must wait for key
                    // build avcC from this keyframe
                    const avcC = buildAvcCFromKeySample(s.data)
                    if (!avcC) {
                        reject(new Error('Failed to synthesize avcC from keyframe'))
                        return
                    }
                    inputAvcC = avcC
                    const decoderLocal = new VideoDecoder({
                        output: frame => onDecodedFrame(frame),
                        error: e => reject(e),
                    })
                    const decCfg: VideoDecoderConfig = { codec: trackInfo!.codec, description: avcC }
                    log('Decoder.configure (synth avcC)', { codec: decCfg.codec, hasDescription: !!decCfg.description })
                    decoderLocal.configure(decCfg)
                    decoder = decoderLocal
                    configured = true
                    seenKey = true // we will feed this key as first chunk below
                }

                if (!decoder) continue // safety

                // From here on, decode normally. Ensure first fed sample is a keyframe.
                if (!seenKey && !s.is_sync) continue
                if (!seenKey && s.is_sync) seenKey = true

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

        // Stream the file into MP4Box in 1 MiB slices
        const u8 = new Uint8Array(fileBuf)
        const pump = () => {
            if (offset >= u8.byteLength) {
                log('All data appended; flushing MP4Box and decoder…')
                mp4.flush()
                if (decoder) {
                    decoder
                        .flush()
                        .then(() => {
                            decoder!.close()
                            resolve()
                        })
                        .catch(reject)
                } else {
                    // No decoder ever configured (e.g., unsupported codec)
                    reject(new Error('Decoder was never configured (unsupported codec or missing avcC)'))
                }
                return
            }
            const end = Math.min(offset + CHUNK, u8.byteLength)
            const slice = u8.subarray(offset, end)
            const ab = slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength) as ArrayBuffer & {
                fileStart?: number
            }
            ;(ab as any).fileStart = offset
            mp4.appendBuffer(ab)
            offset = end
            setTimeout(pump, 0)
        }
        pump()
    })
}

/* ----------------------------- encoder side ------------------------------- */
function createEncoderAndMux(targetW: number, targetH: number, targetFps: number, targetBitrate: number) {
    const canvas = document.createElement('canvas')
    canvas.width = targetW
    canvas.height = targetH
    const ctx = canvas.getContext('2d', { alpha: false })!

    const enc = createEncoder(targetW, targetH, targetFps, targetBitrate)
    let muxer: ReturnType<typeof createMp4Muxer> | null = null

    let nextTsUs = 0
    const frameIntervalUs = Math.round(1e6 / targetFps)

    function onDecodedFrame(frame: VideoFrame) {
        // scale
        drawScaled(ctx, frame, targetW, targetH)

        // synth constant-fps ts
        const outTs = nextTsUs
        nextTsUs += frameIntervalUs

        const outFrame = new VideoFrame(canvas, { timestamp: outTs })
        const key = outTs === 0 || (outTs / 1e6) % 2 === 0
        enc.encoder.encode(outFrame, { keyFrame: key })
        outFrame.close()

        // init muxer on first encoder config
        const cfg = enc.getDecoderConfig()
        if (cfg && !muxer) {
            const desc = cfg.description!
            let avcCOut: ArrayBufferLike
            if (desc instanceof ArrayBuffer) avcCOut = desc
            else if (ArrayBuffer.isView(desc as ArrayBufferView)) {
                const v = desc as ArrayBufferView
                avcCOut = v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength)
            } else {
                throw new Error('Encoder decoderConfig.description missing')
            }
            muxer = createMp4Muxer(targetW, targetH, avcCOut)
            log('Muxer initialized')
        }

        frame.close()
    }

    function finalize(): { chunks: EncodedVideoChunk[]; muxer: ReturnType<typeof createMp4Muxer> } {
        if (!muxer) {
            const cfg = enc.getDecoderConfig()
            if (!cfg) throw new Error('No output decoderConfig for muxer')
            const desc = cfg.description! as ArrayBuffer
            muxer = createMp4Muxer(targetW, targetH, desc)
        }
        return { chunks: enc.chunks, muxer }
    }

    return { onDecodedFrame, finalize, encoder: enc.encoder }
}

/* ------------------------------- main API -------------------------------- */
export async function preprocessVideo(file: File, quality: UploadQuality): Promise<ArrayBuffer> {
    if (!('VideoDecoder' in window) || !('VideoEncoder' in window)) {
        throw new Error('WebCodecs not supported in this browser.')
    }

    const fileBuf = await file.arrayBuffer()
    log('Begin preprocess', { name: file.name, bytes: fileBuf.byteLength, quality })

    // Will be filled in onReady
    let srcTrack: TrackInfo | null = null
    let inputAvcC: ArrayBuffer | null = null
    let targetW = 0,
        targetH = 0,
        targetFps = 0,
        targetBitrate = 0

    const onReadyTrack = (ti: TrackInfo, avcC: ArrayBuffer | null) => {
        srcTrack = ti
        inputAvcC = avcC

        const spec = chooseSpec(ti.width, ti.height, quality)
        const aspect = ti.width / ti.height
        targetW = ti.width >= ti.height ? spec.longSide : Math.round(spec.longSide * aspect)
        targetH = ti.height > ti.width ? spec.longSide : Math.round(spec.longSide / aspect)

        if (targetW % 2) targetW += 1
        if (targetH % 2) targetH += 1

        const srcFpsApprox =
            ti.nb_samples && ti.duration ? Math.max(1, Math.round((ti.nb_samples * ti.timescale) / ti.duration)) : 30
        targetFps = spec.fps || srcFpsApprox
        targetBitrate = spec.bitrate || ti.bitrate || 2_000_000

        log('Source track', ti)
        log('Target spec', { targetW, targetH, targetFps, targetBitrate })
    }

    // Set up encoder + mux-related handlers bound to our targets
    let encMux = createEncoderAndMux(targetW, targetH, targetFps, targetBitrate)
    // but target dimensions are still 0; we'll rebuild encMux once we know them

    const decodedFrames: VideoFrame[] = []
    const onDecodedFrameProxy = (frame: VideoFrame) => {
        // Lazily (re)build enc+mux once we have target sizes
        if (targetW === 0 || targetH === 0 || targetFps === 0) {
            decodedFrames.push(frame) // queue until configured
            return
        }
        if (!encMux || encMux.encoder.state === 'closed') {
            encMux = createEncoderAndMux(targetW, targetH, targetFps, targetBitrate)
            // drain any queued frames
            while (decodedFrames.length) {
                const f = decodedFrames.shift()!
                encMux.onDecodedFrame(f)
            }
        }
        encMux.onDecodedFrame(frame)
    }

    // demux+decode (fast path; will synthesize avcC from first keyframe if missing)
    await decodeAllFramesFast(fileBuf, onReadyTrack, onDecodedFrameProxy)

    // finalize
    if (!encMux) throw new Error('Encoder pipeline missing')
    log('Flushing encoder…')
    await encMux.encoder.flush()
    encMux.encoder.close()

    const { chunks, muxer } = encMux.finalize()
    log(`Muxing ${chunks.length} chunks…`)
    for (const ch of chunks) muxer.addChunk(ch, targetFps)

    const outBuf = muxer.finalize()
    log('Done. Output bytes:', outBuf.byteLength)
    return outBuf
}
