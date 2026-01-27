/**
 * Utilities for loading and rendering a watermark overlay on video or canvas outputs.
 */

/**
 * Represents a loaded watermark image and its intrinsic dimensions.
 */
export type WatermarkAsset = { img: CanvasImageSource; width: number; height: number }

/**
 * Fetches and decodes the default watermark asset (`logo.png`).
 *
 * Uses `ImageBitmap` where available for better performance, falling back to `HTMLImageElement`.
 *
 * @returns A promise resolving to the {@link WatermarkAsset}, or `null` if the asset cannot be loaded or decoded.
 */
export async function getWatermarkAsset(): Promise<WatermarkAsset | null> {
    try {
        // Respect Vite base path if present (e.g. hosted under a subpath).
        const url = new URL(`logo.png`, globalThis.location?.href).toString()

        // Prefer ImageBitmap (works with both canvas types).
        if (typeof fetch === 'function' && typeof createImageBitmap === 'function') {
            const res = await fetch(url, { cache: 'force-cache' })
            if (!res.ok) throw new Error(`Failed to load watermark (${res.status})`)
            const blob = await res.blob()
            const bmp = await createImageBitmap(blob)
            return { img: bmp, width: bmp.width, height: bmp.height }
        }

        // Fallback for environments without createImageBitmap.
        if (typeof document !== 'undefined') {
            const img = new Image()
            img.decoding = 'async'
            img.crossOrigin = 'anonymous'
            img.src = url
            await img.decode()
            return { img, width: img.naturalWidth || img.width, height: img.naturalHeight || img.height }
        }
    } catch {
        // Watermark is best-effort; export should still succeed.
    }
    return null
}

/**
 * Renders the watermark onto a canvas context.
 *
 * The watermark is placed in the bottom-right corner with a subtle shadow and transparency.
 * It automatically scales to fit the output dimensions without upscaling.
 *
 * @param ctx - The canvas rendering context (2D or Offscreen).
 * @param outW - The width of the output canvas.
 * @param outH - The height of the output canvas.
 * @param wm - The watermark asset to draw.
 */
export function drawWatermark(
    ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
    outW: number,
    outH: number,
    wm: WatermarkAsset | null
) {
    if (!wm || wm.width <= 0 || wm.height <= 0) return

    // Keep it unobtrusive: cap size and never upscale.
    const pad = Math.round(outW * 0.02)
    const maxW = Math.min(240, Math.round(outW * 0.22))
    const maxH = Math.min(80, Math.round(outH * 0.18))
    const s = Math.min(maxW / wm.width, maxH / wm.height, 1)
    const w = Math.max(1, Math.round(wm.width * s))
    const h = Math.max(1, Math.round(wm.height * s))

    const x = outW - pad - w
    const y = outH - pad - h

    ctx.save()
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.globalAlpha = 0.65

    // Subtle shadow helps readability on bright backgrounds.
    ctx.shadowColor = 'rgba(0,0,0,0.35)'
    ctx.shadowBlur = 6
    ctx.shadowOffsetX = 0
    ctx.shadowOffsetY = 2

    ctx.drawImage(wm.img, x, y, w, h)
    ctx.restore()
}
