export type BaseRect = { x: number; y: number; w: number; h: number; scale: number }

export function computeBaseRect(outW: number, outH: number, vidW: number, vidH: number): BaseRect {
    const scale = Math.min(outW / vidW, outH / vidH)
    const dispW = vidW * scale
    const dispH = vidH * scale
    const offX = (outW - dispW) / 2
    const offY = (outH - dispH) / 2
    return { x: offX, y: offY, w: dispW, h: dispH, scale }
}
