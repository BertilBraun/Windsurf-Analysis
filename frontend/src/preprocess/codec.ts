type H264LevelNumber = 3.0 | 3.1 | 3.2 | 4.0 | 4.1 | 4.2 | 5.0 | 5.1 | 5.2

/** Map H.264 Level → codec hex (last two bytes of 'avc1.ppccLL') */
const H264_LEVEL_HEX: Record<H264LevelNumber, string> = {
    3.0: '1E',
    3.1: '1F',
    3.2: '20',
    4.0: '28',
    4.1: '29',
    4.2: '2A',
    5.0: '32',
    5.1: '33',
    5.2: '34',
}

/** Compute minimal H.264 level for given frame size & fps (very close approximation). */
function requiredH264Level(codecOptions: CodecOptions): H264LevelNumber {
    const { width, height, fps } = codecOptions
    // MB/s limit drives required level. Macroblock = 16x16.
    const mbW = Math.ceil(width / 16)
    const mbH = Math.ceil(height / 16)
    const mbsPerSec = mbW * mbH * fps

    // Conservative thresholds based on spec tables.
    // (rounded slightly down so we err to the next level up)
    if (mbsPerSec <= 40500) return 3.0 // 720p30-ish
    if (mbsPerSec <= 108000) return 3.1
    if (mbsPerSec <= 216000) return 3.2
    if (mbsPerSec <= 245760) return 4.0
    if (mbsPerSec <= 245760) return 4.1 // same MB/s limit; buffer constraints differ
    if (mbsPerSec <= 522240) return 4.2 // 1080p50/60
    if (mbsPerSec <= 589824) return 5.0 // 4Kp30
    if (mbsPerSec <= 983040) return 5.1 // 4Kp60
    return 5.2 // beyond
}

/** Build H.264 'avc1.ppccLL' for a desired profile & computed level. */
function h264CodecString(profileHex: '42E0' | '4D40' | '6400', level: H264LevelNumber): string {
    return `avc1.${profileHex}${H264_LEVEL_HEX[level]}`
}

/** Additional non-H.264 candidates with broad-ish support (8-bit). */
const VP9_DEFAULT = 'vp09.00.10.08' // profile 0, level 1.0, 8-bit
const AV1_DEFAULT = 'av01.0.08M.08' // Main, 8-bit, MP4-ok parameterization

export interface SelectedCodecInfo {
    codec: string
    config: VideoEncoderConfig
    muxerKind: 'avc' | 'hevc' | 'vp9' | 'av1'
}

/** Map a WebCodecs codec string → mp4-muxer kind ('avc'|'hevc'|'vp9'|'av1'). */
function codecStringToMuxerKind(codec: string): 'avc' | 'hevc' | 'vp9' | 'av1' {
    const c = codec.toLowerCase()
    if (c.startsWith('avc1') || c.startsWith('avc3')) return 'avc'
    if (c.startsWith('hvc1') || c.startsWith('hev1')) return 'hevc'
    if (c.startsWith('vp09')) return 'vp9'
    if (c.startsWith('av01')) return 'av1'
    // Fallback: assume H.264 if WebCodecs accepted something 'avc*'
    throw new Error(`Cannot map codec "${codec}" to mp4-muxer kind`)
}

/** Probe a single config, optionally relaxing hardwareAcceleration on retry. */
async function probeConfig(base: VideoEncoderConfig, relaxHW = true): Promise<VideoEncoderConfig | null> {
    try {
        const sup = await VideoEncoder.isConfigSupported(base)
        if (sup.supported) return sup.config!
    } catch {}
    if (relaxHW && base.hardwareAcceleration) {
        const { hardwareAcceleration, ...rest } = base
        return probeConfig(rest as VideoEncoderConfig, false)
    }
    return null
}

/** Build the ordered codec candidate list given the output geometry and fps. */
function buildCodecCandidates(codecOptions: CodecOptions): string[] {
    const level = requiredH264Level(codecOptions)

    const h264High = h264CodecString('6400', level) // High
    const h264Main = h264CodecString('4D40', level) // Main
    const h264Base = h264CodecString('42E0', level) // Baseline

    const list: string[] = []

    list.push(h264High, h264Main, h264Base, VP9_DEFAULT, AV1_DEFAULT)
    return list
}

export interface CodecOptions {
    width: number
    height: number
    fps: number
    bitrate: number
}

/** Auto-select the first supported codec/config for the given output. */
export async function selectSupportedCodec(options: CodecOptions): Promise<SelectedCodecInfo> {
    const candidates = buildCodecCandidates(options)

    const { width, height, fps, bitrate } = options

    for (const codec of candidates) {
        const cfg: VideoEncoderConfig = {
            codec,
            width,
            height,
            bitrate,
            framerate: fps,
            bitrateMode: 'variable',
            hardwareAcceleration: 'prefer-hardware',
        }
        const supported = await probeConfig(cfg)
        if (supported) return { codec, config: supported, muxerKind: codecStringToMuxerKind(codec) }
    }
    throw new Error(`No supported codec for ${width}x${height}@${fps}fps (tried: ${candidates.join(', ')})`)
}
