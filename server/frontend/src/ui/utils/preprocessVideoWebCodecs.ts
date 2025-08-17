export async function preprocessVideo(file: File): Promise<{ arrayBuffer: ArrayBuffer; type: string; name: string }> {
    if (!('MediaRecorder' in window) || !('captureStream' in HTMLCanvasElement.prototype)) {
        // Fallback: passthrough
        const arrayBuffer = await file.arrayBuffer()
        return { arrayBuffer, type: file.type || 'video/webm', name: file.name }
    }

    const video = document.createElement('video')
    video.preload = 'auto'
    video.muted = true
    video.playsInline = true

    const src = URL.createObjectURL(file)
    video.src = src

    try {
        await new Promise<void>((resolve, reject) => {
            const onLoaded = () => resolve()
            const onError = () => reject(new Error('Failed to load video'))
            video.addEventListener('loadedmetadata', onLoaded, { once: true })
            video.addEventListener('error', onError, { once: true })
        })
    } catch {
        // Browser cannot decode this file/codec; skip transcoding and passthrough
        URL.revokeObjectURL(src)
        const arrayBuffer = await file.arrayBuffer()
        return { arrayBuffer, type: file.type || 'application/octet-stream', name: file.name }
    }

    const sourceWidth = Math.max(1, video.videoWidth || 0)
    const sourceHeight = Math.max(1, video.videoHeight || 0)

    const targetAspect = 16 / 9
    let cropSx = 0
    let cropSy = 0
    let cropSw = sourceWidth
    let cropSh = sourceHeight
    let canvasWidth = sourceWidth
    let canvasHeight = sourceHeight

    if (sourceWidth > 1920 || sourceHeight > 1080) {
        // Center-crop to 16:9 then scale to 1920x1080
        const srcAspect = sourceWidth / sourceHeight
        if (srcAspect > targetAspect) {
            // Too wide: crop width
            cropSh = sourceHeight
            cropSw = Math.round(sourceHeight * targetAspect)
            cropSx = Math.floor((sourceWidth - cropSw) / 2)
            cropSy = 0
        } else {
            // Too tall: crop height
            cropSw = sourceWidth
            cropSh = Math.round(sourceWidth / targetAspect)
            cropSx = 0
            cropSy = Math.floor((sourceHeight - cropSh) / 2)
        }
        canvasWidth = 1920
        canvasHeight = 1080
    }

    const canvas = document.createElement('canvas')
    canvas.width = canvasWidth
    canvas.height = canvasHeight
    const ctx = canvas.getContext('2d', { alpha: false })!

    // Prefer VP9, fallback to VP8
    const mimeCandidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
    const mimeType = mimeCandidates.find(t => (window as any).MediaRecorder.isTypeSupported?.(t)) || 'video/webm'

    const stream = (canvas as any).captureStream?.(25) as MediaStream
    if (!stream) {
        const arrayBuffer = await file.arrayBuffer()
        return { arrayBuffer, type: file.type || 'video/webm', name: file.name }
    }

    const chunks: Blob[] = []
    const recorder = new MediaRecorder(stream, { mimeType })
    recorder.ondataavailable = e => {
        if (e.data && e.data.size > 0) chunks.push(e.data)
    }

    const pumpFrames = async (): Promise<void> => {
        return new Promise<void>(resolve => {
            let lastCapturedMediaTime = -1
            const minDelta = 1 / 25 // cap at 25 fps; if original < 25, frames naturally arrive slower
            const handle = (now: number, meta: any) => {
                const mt = meta?.mediaTime ?? video.currentTime
                if (lastCapturedMediaTime < 0 || mt - lastCapturedMediaTime >= minDelta) {
                    ctx.drawImage(video, cropSx, cropSy, cropSw, cropSh, 0, 0, canvasWidth, canvasHeight)
                    // Ensure a frame is emitted on the stream
                    const tracks = stream.getVideoTracks()
                    if (tracks[0] && (tracks[0] as any).requestFrame) (tracks[0] as any).requestFrame()
                    lastCapturedMediaTime = mt
                }
                if (!video.ended && mt < (video.duration || Infinity)) {
                    ;(video as any).requestVideoFrameCallback?.(handle)
                } else {
                    resolve()
                }
            }
            ;(video as any).requestVideoFrameCallback?.(handle)
        })
    }

    await video.play().catch(() => video.pause())
    recorder.start()
    await pumpFrames()
    recorder.stop()

    await new Promise(resolve => (recorder.onstop = resolve))

    URL.revokeObjectURL(src)

    const blob = new Blob(chunks, { type: mimeType })
    const arrayBuffer = await blob.arrayBuffer()
    const base = file.name.replace(/\.[^/.]+$/, '')
    const ext = mimeType.includes('vp9') || mimeType.includes('vp8') ? 'webm' : 'webm'
    return { arrayBuffer, type: blob.type || 'video/webm', name: `${base}.${ext}` }
}
