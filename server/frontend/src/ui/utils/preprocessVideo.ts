// src/ui/utils/preprocessVideo.ts
/* eslint-disable no-console */

import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile } from '@ffmpeg/util'

export type UploadQuality = 'original' | 'high' | 'medium' | 'minimum'

interface TargetSpec {
    longSide: number
    fps: number | null
}

let ffmpegSingleton: FFmpeg | null = null
async function getFFmpeg(): Promise<FFmpeg> {
    if (!ffmpegSingleton) {
        ffmpegSingleton = new FFmpeg()
        await ffmpegSingleton.load()
    } else if (!ffmpegSingleton.loaded) {
        await ffmpegSingleton.load()
    }
    return ffmpegSingleton
}

function chooseSpec(srcW: number, srcH: number, q: UploadQuality): TargetSpec {
    const longer = Math.max(srcW, srcH)
    if (q === 'original') return { longSide: longer, fps: null }
    if (q === 'high') return { longSide: longer >= 1920 ? 1920 : longer >= 1280 ? 1280 : 640, fps: 30 }
    if (q === 'medium') return { longSide: longer >= 1280 ? 1280 : 640, fps: 25 }
    return { longSide: 640, fps: 15 }
}

// Probe dimensions using a <video> tag (no ffprobe in wasm)
async function probeVideo(file: File): Promise<{ width: number; height: number }> {
    const url = URL.createObjectURL(file)
    try {
        const v = document.createElement('video')
        v.preload = 'metadata'
        v.src = url
        await new Promise<void>(res => {
            if (v.readyState >= 1) res()
            else v.onloadedmetadata = () => res()
        })
        return { width: v.videoWidth || 1, height: v.videoHeight || 1 }
    } finally {
        URL.revokeObjectURL(url)
    }
}

// Fit inside longSide, preserve AR, force even dims
function buildScaleFilter(longSide: number) {
    return `scale='if(gte(iw,ih),${longSide},-2)':'if(gte(ih,iw),${longSide},-2)':flags=bicubic:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2`
}

// Try a tiny dry run to detect if an encoder exists
async function encoderAvailable(ffmpeg: FFmpeg, name: string): Promise<boolean> {
    try {
        await ffmpeg.exec(['-hide_banner', '-loglevel', 'error', '-h', `encoder=${name}`])
        return true
    } catch {
        console.log(`[preprocessVideo] encoder ${name} not available`)
        return false
    }
}

function throwWithLogs(logs: string[], msg: string): never {
    console.error('[ffmpeg] logs:\n' + logs.join('\n'))
    throw new Error(msg)
}

export async function preprocessVideo(file: File, quality: UploadQuality): Promise<ArrayBuffer> {
    console.log('[preprocessVideo] start', { name: file.name, size: file.size, quality })

    const { width: srcW, height: srcH } = await probeVideo(file)
    const spec = chooseSpec(srcW, srcH, quality)
    console.log('[preprocessVideo] source', { srcW, srcH }, 'spec', spec)

    const ffmpeg = await getFFmpeg()
    console.log('[preprocessVideo] ffmpeg loaded')
    const logs: string[] = []
    ffmpeg.on('log', ({ message }) => logs.push(message))

    const inName = 'in.mp4'
    // prefer mp4 when we can (x264); otherwise webm (vp9)
    let outName = quality === 'original' ? 'out.mp4' : 'out.mp4'

    // clean workspace
    try {
        ffmpeg.deleteFile(inName)
    } catch {
        console.log(`[preprocessVideo] failed to delete file ${inName}`)
    }
    try {
        ffmpeg.deleteFile('out.mp4')
    } catch {
        console.log(`[preprocessVideo] failed to delete file out.mp4`)
    }
    try {
        ffmpeg.deleteFile('out.webm')
    } catch {
        console.log(`[preprocessVideo] failed to delete file out.webm`)
    }

    console.log('[preprocessVideo] writing file to ffmpeg', file.name)
    const buf = await fetchFile(file)
    console.log('[preprocessVideo] file fetched', buf.byteLength)
    ffmpeg.writeFile(inName, buf)

    // Pick encoder
    const hasX264 = await encoderAvailable(ffmpeg, 'libx264')
    const useX264 = hasX264 // prefer x264
    if (!useX264) outName = 'out.webm'

    const vf = spec.fps === null ? null : buildScaleFilter(spec.longSide)

    // Build args
    const args: string[] = ['-y', '-i', inName, '-an']

    if (useX264) {
        // H.264 — faster than VP9 in wasm; use a very fast preset to avoid stalls
        // If quality "original": no -vf and no -r (keep size/fps)
        args.push(
            '-c:v',
            'libx264',
            '-preset',
            'veryfast',
            '-tune',
            'zerolatency',
            '-crf',
            '23',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart'
        )
        if (vf) {
            args.push('-vf', `format=yuv420p,${vf}`)
            args.push('-r', String(spec.fps))
        }
        // else keep source fps/size
        args.push(outName)
    } else {
        // VP9 fallback (slower). Make it as fast as possible.
        args.push(
            '-c:v',
            'libvpx-vp9',
            '-b:v',
            '0',
            '-crf',
            '32',
            '-deadline',
            'realtime',
            '-cpu-used',
            '8',
            '-row-mt',
            '1',
            '-threads',
            '1'
        )
        if (vf) {
            args.push('-vf', vf, '-r', String(spec.fps))
        }
        outName = 'out.webm'
        args.push(outName)
    }

    console.log('[preprocessVideo] ffmpeg args', args.join(' '))

    try {
        await ffmpeg.exec(args)
    } catch (e) {
        return throwWithLogs(logs, 'ffmpeg.wasm failed to run (see logs above).')
    }

    console.log('[preprocessVideo] ffmpeg exec done')

    // read output
    let data: Uint8Array
    try {
        data = await ffmpeg.readFile(outName)
    } catch {
        return throwWithLogs(logs, `Output file ${outName} not found (command likely failed).`)
    }

    // cleanup
    try {
        ffmpeg.deleteFile(inName)
    } catch {
        console.log(`[preprocessVideo] failed to delete file ${inName}`)
    }
    try {
        ffmpeg.deleteFile(outName)
    } catch {
        console.log(`[preprocessVideo] failed to delete file ${outName}`)
    }

    console.log('[preprocessVideo] done', { bytes: data.length, encoder: useX264 ? 'libx264' : 'libvpx-vp9' })
    return data.buffer as ArrayBuffer
}
