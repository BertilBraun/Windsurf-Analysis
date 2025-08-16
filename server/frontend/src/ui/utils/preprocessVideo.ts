import { FFmpeg } from '@ffmpeg/ffmpeg'
import { toBlobURL } from '@ffmpeg/util'

export async function preprocessVideo(file: File): Promise<{ arrayBuffer: ArrayBuffer; type: string; name: string }> {
    // Lazy-init a singleton ffmpeg instance (module-level cache shared via window)
    let ffmpegInstance: FFmpeg | null = (window as any).__ffmpegSingleton || null
    let ffmpegLoading: Promise<FFmpeg> | null = (window as any).__ffmpegLoading || null
    const getFfmpeg = async (): Promise<FFmpeg> => {
        if (ffmpegInstance) return ffmpegInstance
        if (!ffmpegLoading) {
            const instance = new FFmpeg()
            // Load core (multi-thread build) from CDN to avoid bundler asset config
            const baseURL = 'https://unpkg.com/@ffmpeg/core-mt@0.12.6/dist/esm'
            ffmpegLoading = instance
                .load({
                    coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
                    wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
                    workerURL: await toBlobURL(`${baseURL}/ffmpeg-core.worker.js`, 'text/javascript'),
                })
                .then(() => instance)
                ; (window as any).__ffmpegLoading = ffmpegLoading
        }
        ffmpegInstance = await ffmpegLoading
        ;(window as any).__ffmpegSingleton = ffmpegInstance
        return ffmpegInstance
    }

    // Helper to get intrinsic width/height via HTMLVideoElement
    const getElementDimensions = (blobUrl: string): Promise<{ width: number; height: number }> =>
        new Promise((resolve, reject) => {
            const vid = document.createElement('video')
            const cleanup = () => {
                URL.revokeObjectURL(blobUrl)
                vid.removeAttribute('src')
                try {
                    vid.load()
                } catch {}
            }
            vid.preload = 'metadata'
            vid.onloadedmetadata = () => {
                const width = vid.videoWidth || 0
                const height = vid.videoHeight || 0
                cleanup()
                resolve({ width, height })
            }
            vid.onerror = () => {
                cleanup()
                reject(new Error('Failed to read video metadata'))
            }
            vid.src = blobUrl
        })

    // Fallback passthrough helper
    const passthrough = async () => {
        const arrayBuffer = await file.arrayBuffer()
        return { arrayBuffer, type: file.type || 'video/mp4', name: file.name }
    }

    try {
        const ffmpeg = await getFfmpeg()

        // Write input
        const inputName = 'input'
        const inputArray = new Uint8Array(await file.arrayBuffer())
        await ffmpeg.writeFile(inputName, inputArray)

        // Probe FPS from ffmpeg logs
        let detectedFps: number | null = null
        const logHandler = ({ message }: { type: string; message: string }) => {
            const fpsMatch = message.match(/,\s*(\d+(?:\.\d+)?)\s*fps\b/i)
            const tbrMatch = message.match(/,\s*(\d+(?:\.\d+)?)\s*tbr\b/i)
            if (!detectedFps) {
                const val = fpsMatch ? parseFloat(fpsMatch[1]) : tbrMatch ? parseFloat(tbrMatch[1]) : null
                if (val && Number.isFinite(val)) detectedFps = val
            }
        }
        ffmpeg.on('log', logHandler)
        try {
            await ffmpeg.exec(['-hide_banner', '-i', inputName, '-f', 'null', '-'])
        } catch {
            // Expected non-zero exit due to no output, but logs contain metadata
        }

        // Get dimensions via element (more reliable for width/height)
        const blobUrl = URL.createObjectURL(file)
        const { width: inW, height: inH } = await getElementDimensions(blobUrl)

        const needsResize = inW > 1920 || inH > 1080
        const needsFpsLimit = detectedFps !== null && detectedFps > 25

        const outputName = 'output.mp4'
        const args: string[] = ['-y', '-i', inputName, '-an'] // remove audio

        if (needsResize) {
            // Scale to fit within 1920x1080 then crop to 1920x1080 to enforce exact frame size without upscaling
            args.push('-vf', "scale='if(gt(a,16/9),-2,1920)':'if(gt(a,16/9),1080,-2)',crop=1920:1080")
        }

        if (needsFpsLimit) {
            args.push('-r', '25')
        }

        // Encode to MP4 (H.264). If libx264 is unavailable in this build, ffmpeg will throw and we'll fall back to passthrough.
        args.push('-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', outputName)

        await ffmpeg.exec(args)
        const data = (await ffmpeg.readFile(outputName)) as Uint8Array

        const arrayBuffer = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)

        // Name output with .mp4 extension
        const base = file.name.replace(/\.[^/.]+$/, '')
        return { arrayBuffer, type: 'video/mp4', name: `${base}.mp4` }
    } catch (err) {
        // As a fail-safe, return the original file bytes
        alert('Failed to preprocess video: ' + file.name + '\n\n' + err)
        return await passthrough()
    }
}
