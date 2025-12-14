export function quantizeOrientation(deg: number): 0 | 90 | 180 | 270 {
    const r = (-Math.round(deg) + 360) % 360
    if (r >= 315 || r < 45) return 0
    if (r >= 45 && r < 135) return 90
    if (r >= 135 && r < 225) return 180
    return 270
}

export function getRotatedDimensions(width: number, height: number, deg: number): { width: number; height: number } {
    const rot = quantizeOrientation(deg)
    if (rot === 90 || rot === 270) return { width: height, height: width }
    return { width, height }
}

export function drawRotatedToCanvas(
    frame: VideoFrame | HTMLVideoElement,
    target: HTMLCanvasElement,
    deg: number
): { width: number; height: number } {
    if (frame instanceof HTMLVideoElement) {
        var srcW = frame.videoWidth
        var srcH = frame.videoHeight
    } else {
        var srcW = frame.displayWidth || frame.codedWidth
        var srcH = frame.displayHeight || frame.codedHeight
    }
    const rot = quantizeOrientation(deg)
    let outW = srcW
    let outH = srcH
    if (rot === 90 || rot === 270) {
        outW = srcH
        outH = srcW
    }

    if (target.width !== outW || target.height !== outH) {
        target.width = outW
        target.height = outH
    }
    const ctx = target.getContext('2d')!
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    if (rot === 90) {
        ctx.translate(outW, 0)
        ctx.rotate(Math.PI / 2)
    } else if (rot === 180) {
        ctx.translate(outW, outH)
        ctx.rotate(Math.PI)
    } else if (rot === 270) {
        ctx.translate(0, outH)
        ctx.rotate((3 * Math.PI) / 2)
    }
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(frame, 0, 0)
    return { width: outW, height: outH }
}
