import MP4Box, { MP4File, MP4ArrayBuffer } from 'mp4box'
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

/**
 * Preprocess video client-side using WebCodecs + MP4Box.js
 */
export async function preprocessVideo(file: File, quality: UploadQuality): Promise<ArrayBuffer> {
    const buf = await file.arrayBuffer()

    // --- Step 1: Parse MP4 container with MP4Box.js ---
    const mp4boxfile: MP4File = MP4Box.createFile()
    const sourceBuffer = buf as MP4ArrayBuffer
    ;(sourceBuffer as any).fileStart = 0
    mp4boxfile.appendBuffer(sourceBuffer)
    mp4boxfile.flush()

    // Pick first video track
    const videoTrack = mp4boxfile.getInfo().tracks.find(t => t.video)
    if (!videoTrack) throw new Error('No video track found')
    const { width, height, timescale, id: trackId, duration } = videoTrack

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

    const encCfg: VideoEncoderConfig = {
        codec: 'avc1.42E01E', // H.264 baseline
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

    // --- Step 3: Decode frames fast ---
    const decoder = new VideoDecoder({
        output: (frame: VideoFrame) => {
            // draw scaled frame
            ctx.drawImage(frame, 0, 0, targetW, targetH)

            // Wrap canvas back into VideoFrame with synthetic timestamp
            const vf = new VideoFrame(canvas, { timestamp: frame.timestamp })
            encoder.encode(vf, { keyFrame: frame.timestamp === 0 })
            vf.close()
            frame.close()
        },
        error: e => console.error('Decoder error:', e),
    })

    decoder.configure({ codec: videoTrack.codec })

    // Feed samples into decoder
    mp4boxfile.onSamples = (id, _user, samples) => {
        if (id !== trackId) return
        for (const s of samples) {
            const chunk = new EncodedVideoChunk({
                type: s.is_sync ? 'key' : 'delta',
                timestamp: s.dts, // microseconds if timescale = 1e6
                duration: s.duration,
                data: new Uint8Array(s.data),
            })
            decoder.decode(chunk)
        }
    }

    // Ask MP4Box to extract all samples
    mp4boxfile.setExtractionOptions(trackId)
    mp4boxfile.start()

    // Wait until decode/encode finishes
    await decoder.flush()
    await encoder.flush()
    encoder.close()
    decoder.close()

    // --- Step 4: Mux encoded chunks back into MP4 ---
    if (!decoderConfig) throw new Error('No decoderConfig from encoder')

    const muxer = MP4Box.createFile()
    const avcC = new Uint8Array(decoderConfig.description!)
    const outTrackId = muxer.addTrack({
        timescale: 1_000_000,
        width: targetW,
        height: targetH,
        avcDecoderConfigRecord: avcC.buffer,
        hdlr: 'vide',
    })
    muxer.setSegmentOptions(outTrackId, null, { nbSamples: chunks.length })
    muxer.initializeSegmentation()
    muxer.start()

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
        ;(u8.buffer as any).fileStart = 0
        muxer.addSample(outTrackId, u8.buffer, sample)
    }

    muxer.flush()

    const segs: ArrayBuffer[] = []
    muxer.onSegment = (_id, _user, buf: ArrayBuffer) => segs.push(buf)

    // Concatenate segments
    let total = segs.reduce((s, b) => s + b.byteLength, 0)
    const out = new Uint8Array(total)
    let offset = 0
    for (const b of segs) {
        out.set(new Uint8Array(b), offset)
        offset += b.byteLength
    }

    return out.buffer
}
