import * as MP4Box from 'mp4box'
import { UploadQuality } from '../types'

interface TargetSpec {
    longSide: number
    fps: number
    bitrate: number
}

/**
 * Decide target resolution & fps based on quality + source size.
 */
function chooseSpec(width: number, height: number, quality: UploadQuality): TargetSpec {
    const longer = Math.max(width, height)

    const bitrate = {
        high: 4_000_000,
        medium: 2_000_000,
        minimum: 800_000,
    }

    if (quality === 'original') {
        return { longSide: longer, fps: 0, bitrate: 0 } // fps=0 means "keep original"
    }
    if (quality === 'high') {
        if (longer >= 1920) return { longSide: 1920, fps: 30, bitrate: bitrate.high }
        if (longer >= 1280) return { longSide: 1280, fps: 30, bitrate: bitrate.high }
        return { longSide: 640, fps: 30, bitrate: bitrate.high }
    }
    if (quality === 'medium') {
        return { longSide: longer >= 1280 ? 1280 : 640, fps: 25, bitrate: bitrate.medium }
    }
    if (quality === 'minimum') {
        return { longSide: 640, fps: 15, bitrate: bitrate.minimum }
    }
    throw new Error('Unknown quality ' + quality)
}

function chooseAvcLevelHex(width: number, height: number, fps: number): string {
    const area = width * height
    // Conservative thresholds by coded area; fps considered typical (<=30)
    // L3.0 (0x1E): up to ~414,720 (e.g., 720x576)
    // L3.1 (0x1F): up to ~921,600 (e.g., 1280x720)
    // L4.0 (0x28): up to ~2,073,600 (e.g., 1920x1080)
    if (area <= 414_720) return '1E' // 3.0
    if (area <= 921_600) return '1F' // 3.1
    if (area <= 2_073_600) return '28' // 4.0
    return '29' // 4.1 as a safe upper fallback
}

/**
 * Preprocess video client-side using WebCodecs + MP4Box.js
 */
export async function preprocessVideo(file: File, quality: UploadQuality): Promise<ArrayBuffer> {
    const buf = await file.arrayBuffer()

    // --- Step 1: Parse MP4 container with MP4Box.js ---
    const mp4boxfile = MP4Box.createFile()
    // Set onReady before appending to ensure track info is available
    let info: ReturnType<typeof mp4boxfile.getInfo> | null = null
    await new Promise<void>((resolve, reject) => {
        mp4boxfile.onError = () => reject(new Error('MP4Box error while parsing'))
        mp4boxfile.onReady = _info => {
            info = _info
            resolve()
        }
        const sourceBuffer = buf as MP4Box.MP4BoxBuffer
        sourceBuffer.fileStart = 0
        mp4boxfile.appendBuffer(sourceBuffer)
    })

    // Pick first video track
    const videoTrack = (info || mp4boxfile.getInfo()).tracks.find(t => t.video)
    if (!videoTrack) throw new Error('No video track found')
    const { video, timescale, id: trackId, duration } = videoTrack

    const width = video?.width ?? 1
    const height = video?.height ?? 1
    const spec = chooseSpec(width, height, quality)

    // Compute target width/height preserving AR
    const aspect = width / height
    const targetW = width >= height ? spec.longSide : Math.round(spec.longSide * aspect)
    const targetH = height > width ? spec.longSide : Math.round(spec.longSide / aspect)

    // Target fps: if 0 (original), derive from track
    const targetFps = spec.fps || Math.round((videoTrack.nb_samples || 0) / (duration / timescale))
    // Target bitrate: if 0 (original), derive from track
    const targetBitrate = spec.bitrate || videoTrack.bitrate

    console.log(
        'Processing video',
        file.name,
        'from',
        width,
        'x',
        height,
        'to',
        targetW,
        'x',
        targetH,
        '@',
        targetFps,
        'fps with',
        targetBitrate,
        'bitrate'
    )

    // --- Step 2: Set up encoder + muxer ---
    const chunks: EncodedVideoChunk[] = []
    let decoderConfig: VideoDecoderConfig | undefined

    const encoder = new VideoEncoder({
        output: (chunk, meta) => {
            if (meta?.decoderConfig && !decoderConfig) {
                decoderConfig = meta.decoderConfig
            }
            chunks.push(chunk)
        },
        error: e => console.error('Encoder error:', e),
    })

    const levelHex = chooseAvcLevelHex(targetW, targetH, targetFps)
    const encCfg: VideoEncoderConfig = {
        codec: `avc1.42E0${levelHex}`, // H.264 Baseline, dynamic level
        width: targetW,
        height: targetH,
        framerate: targetFps,
        bitrate: targetBitrate,
        hardwareAcceleration: 'prefer-hardware',
    }
    encoder.configure(encCfg)

    // Prepare canvas for rescaling
    const canvas = document.createElement('canvas')
    canvas.width = targetW
    canvas.height = targetH
    const ctx = canvas.getContext('2d', { alpha: false })!

    console.log('Encoder config:', encCfg)
    console.log('Decoder config:', { codec: videoTrack.codec })

    // --- Step 3: Decode frames fast ---
    let firstEncoded = true
    const decoder = new VideoDecoder({
        output: (frame: VideoFrame) => {
            // draw scaled frame
            ctx.drawImage(frame, 0, 0, targetW, targetH)

            // Wrap canvas back into VideoFrame with synthetic timestamp
            const vf = new VideoFrame(canvas, { timestamp: frame.timestamp })
            const makeKey = firstEncoded
            encoder.encode(vf, { keyFrame: makeKey })
            if (firstEncoded) firstEncoded = false
            vf.close()
            frame.close()
        },
        error: e => console.error('Decoder error:', e),
    })

    decoder.configure({ codec: videoTrack.codec })

    // Some encoders emit decoderConfig only after a keyframe flush; queue a forced keyframe at start
    // by encoding a zero-area dummy frame if no real frames arrive quickly.
    // In practice, we rely on first frame keyframe above.

    console.log('Feeding samples into decoder')

    // Feed samples into decoder and await completion using a dedicated extractor instance
    const totalSamples = videoTrack.nb_samples || 0
    let processedSamples = 0

    await new Promise<void>((resolve, reject) => {
        const extractor = MP4Box.createFile()
        extractor.onError = () => reject(new Error('MP4Box error during extraction'))
        extractor.onSamples = (id, _user, samples) => {
            if (id !== trackId) return
            console.log('Received samples', samples.length)
            for (const s of samples) {
                if (!s.data) throw new Error('No data in sample')
                const tsUs = Math.round((s.dts / timescale) * 1_000_000)
                const durUs = s.duration ? Math.round((s.duration / timescale) * 1_000_000) : undefined
                const chunk = new EncodedVideoChunk({
                    type: s.is_sync ? 'key' : 'delta',
                    timestamp: tsUs,
                    duration: durUs,
                    data: new Uint8Array(s.data),
                })
                decoder.decode(chunk)
                processedSamples += 1
            }
            if (totalSamples > 0 && processedSamples >= totalSamples) {
                resolve()
            }
        }
        // Configure extraction BEFORE appending
        if (totalSamples > 0) {
            extractor.setExtractionOptions(trackId, null, { nbSamples: totalSamples })
        } else {
            extractor.setExtractionOptions(trackId)
        }
        extractor.start()
        // Append the entire file in chunks so extractor can emit samples while parsing
        const chunkSize = 1 * 1024 * 1024
        let offset = 0
        while (offset < buf.byteLength) {
            const end = Math.min(offset + chunkSize, buf.byteLength)
            const chunk = buf.slice(offset, end) as MP4Box.MP4BoxBuffer
            chunk.fileStart = offset
            extractor.appendBuffer(chunk)
            offset = end
        }
        extractor.flush()
        if (totalSamples === 0) {
            // If totalSamples unknown, resolve after flush completes.
            // onSamples will have been called as needed during parsing.
            resolve()
        }
    })

    console.log('Decoder flushing')
    // Now wait until decode/encode finishes
    await decoder.flush()
    await encoder.flush()
    encoder.close()
    decoder.close()

    // --- Step 4: Mux encoded chunks back into MP4 ---
    if (!decoderConfig) {
        // As a fallback, try to derive minimal AVC config from encoder config if available
        // Some implementations delay decoderConfig; in that case, abort gracefully
        throw new Error('No decoderConfig from encoder')
    }

    console.log('Muxing encoded chunks back into MP4')

    const muxer = MP4Box.createFile()
    // Normalize description (AllowSharedBufferSource) into a Uint8Array
    const desc = decoderConfig.description!
    let avcC: Uint8Array
    if (desc instanceof ArrayBuffer) {
        avcC = new Uint8Array(desc)
    } else if (ArrayBuffer.isView(desc as ArrayBufferView)) {
        const view = desc as ArrayBufferView
        avcC = new Uint8Array(view.buffer, view.byteOffset, view.byteLength)
    } else {
        avcC = new Uint8Array(0)
    }
    const outTrackId = muxer.addTrack({
        timescale: 1_000_000,
        width: targetW,
        height: targetH,
        avcDecoderConfigRecord:
            avcC.byteOffset === 0 && avcC.byteLength === avcC.buffer.byteLength ? avcC.buffer : avcC.slice().buffer,
        hdlr: 'vide',
    })
    // Prepare to collect segments BEFORE starting segmentation
    const segs: ArrayBuffer[] = []
    muxer.onSegment = (_id, _user, buffer /* ArrayBuffer */) => {
        segs.push(buffer)
    }
    muxer.setSegmentOptions(outTrackId, null, { nbSamples: chunks.length })
    muxer.initializeSegmentation()
    muxer.start()

    console.log('Adding samples to muxer')

    for (const ch of chunks) {
        const u8 = new Uint8Array(ch.byteLength)
        ch.copyTo(u8)
        const sample = {
            duration: ch.duration ?? Math.round(1e6 / targetFps),
            dts: ch.timestamp,
            cts: 0,
            is_sync: ch.type === 'key',
            size: u8.byteLength,
        }
        muxer.addSample(outTrackId, u8, sample)
    }

    muxer.flush()
    console.log('Muxer flushed')

    // Concatenate segments
    let total = segs.reduce((s, b) => s + b.byteLength, 0)
    const out = new Uint8Array(total)
    let offset = 0
    for (const b of segs) {
        out.set(new Uint8Array(b), offset)
        offset += b.byteLength
    }

    console.log('Done')

    return out.buffer
}
