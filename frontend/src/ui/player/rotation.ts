/**
 * Utilities for handling rotation of video frames and canvas elements.
 */

/**
 * Normalizes and snaps a rotation angle to the nearest 90-degree increment.
 * @param deg Rotation in degrees.
 * @returns The quantized angle (0, 90, 180, or 270).
 */
export function quantizeOrientation(deg: number): 0 | 90 | 180 | 270 {
    const r = (-Math.round(deg) + 360) % 360
    if (r >= 315 || r < 45) return 0
    if (r >= 45 && r < 135) return 90
    if (r >= 135 && r < 225) return 180
    return 270
}

/**
 * Calculates the dimensions of a rectangle after applying a quantized rotation.
 * @param width Original width.
 * @param height Original height.
 * @param deg Rotation in degrees.
 * @returns The rotated width and height.
 */
export function getRotatedDimensions(width: number, height: number, deg: number): { width: number; height: number } {
    const rot = quantizeOrientation(deg)
    if (rot === 90 || rot === 270) return { width: height, height: width }
    return { width, height }
}

/**
 * Draws a source frame to a canvas with the specified rotation.
 * Automatically resizes the target canvas and applies necessary 2D transformations.
 * @param frame The source image, video, or frame to draw.
 * @param target The canvas element to draw into.
 * @param deg Rotation in degrees.
 * @param explicitSize Optional override for source dimensions.
 * @returns The dimensions of the resulting output on the canvas.
 */
export function drawRotatedToCanvas(
    frame: VideoFrame | HTMLVideoElement | CanvasImageSource,
    target: HTMLCanvasElement,
    deg: number,
    explicitSize?: { width: number; height: number }
): { width: number; height: number } {
    let srcW: number
    let srcH: number
    if (explicitSize && explicitSize.width > 0 && explicitSize.height > 0) {
        srcW = explicitSize.width
        srcH = explicitSize.height
    } else if (frame instanceof HTMLVideoElement) {
        srcW = frame.videoWidth
        srcH = frame.videoHeight
    } else if (typeof VideoFrame !== 'undefined' && frame instanceof VideoFrame) {
        srcW = frame.displayWidth || frame.codedWidth
        srcH = frame.displayHeight || frame.codedHeight
    } else if (frame instanceof HTMLCanvasElement) {
        srcW = frame.width
        srcH = frame.height
    } else if (typeof OffscreenCanvas !== 'undefined' && frame instanceof OffscreenCanvas) {
        srcW = frame.width
        srcH = frame.height
    } else if (typeof ImageBitmap !== 'undefined' && frame instanceof ImageBitmap) {
        srcW = frame.width
        srcH = frame.height
    } else {
        srcW = (frame as any).width ?? 0
        srcH = (frame as any).height ?? 0
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
    ctx.drawImage(frame as any, 0, 0)
    return { width: outW, height: outH }
}
