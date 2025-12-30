import { clamp } from '../utils/clamp'
import { MAX_SCALE, MIN_SCALE, TARGET_BBOX_HEIGHT_RATIO } from './constants'
import { computeBaseRect } from './renderMath'
import { drawRotatedToCanvas, getRotatedDimensions } from './rotation'
import { PlayerState } from './state'

export type OverviewView = {
    zoom: number
    detailedZoom?: number
    offsetX: number
    offsetY: number
    hoveredTrackId: number | null
}

export type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D
export type TimedBBox = { time_percent: number; bbox: [number, number, number, number] }
export type AnnotationPoint = { x: number; y: number }
export type AnnotationStroke = {
    id: string
    frameIndex: number
    color: string
    width: number
    points: AnnotationPoint[]
}

let _sharedOffscreenCanvas: HTMLCanvasElement | null = null
export function getSharedOffscreenCanvas(): HTMLCanvasElement {
    if (!_sharedOffscreenCanvas) {
        _sharedOffscreenCanvas = document.createElement('canvas')
    }
    return _sharedOffscreenCanvas
}

export function ensureCanvasSize(canvas: HTMLCanvasElement, cssWidth: number, cssHeight: number) {
    const dpr = Math.max(1, Math.floor(window.devicePixelRatio))
    const needW = Math.max(1, Math.floor(cssWidth * dpr))
    const needH = Math.max(1, Math.floor(cssHeight * dpr))
    if (canvas.width !== needW || canvas.height !== needH) {
        canvas.width = needW
        canvas.height = needH
    }
    canvas.style.width = `${cssWidth}px`
    canvas.style.height = `${cssHeight}px`
    const ctx = canvas.getContext('2d')!
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.scale(dpr, dpr)
    return ctx
}

function drawFitContain(ctx: Ctx2D, outW: number, outH: number, src: CanvasImageSource, srcW: number, srcH: number) {
    const base = computeBaseRect(outW, outH, srcW, srcH)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(src, 0, 0, srcW, srcH, base.x, base.y, base.w, base.h)
}

export function drawDetailedCrop(
    ctx: Ctx2D,
    outputWidth: number,
    outputHeight: number,
    srcCanvas: CanvasImageSource,
    srcWidth: number,
    srcHeight: number,
    det: TimedBBox | null,
    zoomMul: number = 1
) {
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, outputWidth, outputHeight)

    if (!det) {
        drawFitContain(ctx, outputWidth, outputHeight, srcCanvas, srcWidth, srcHeight)
        return
    }

    const [x1p, y1p, x2p, y2p] = det.bbox
    const x1 = x1p * srcWidth
    const y1 = y1p * srcHeight
    const x2 = x2p * srcWidth
    const y2 = y2p * srcHeight
    const bboxW = Math.max(1, x2 - x1)
    const bboxH = Math.max(1, y2 - y1)

    const sHeight = (TARGET_BBOX_HEIGHT_RATIO * outputHeight) / bboxH
    const sWidthLimit = outputWidth / bboxW
    const sBase = clamp(Math.min(sHeight, sWidthLimit), MIN_SCALE, MAX_SCALE)
    const s = clamp(sBase * Math.max(1e-6, zoomMul), MIN_SCALE, MAX_SCALE)

    const cx = (x1 + x2) * 0.5
    const cy = (y1 + y2) * 0.5
    const cropW = outputWidth / s
    const cropH = outputHeight / s
    const winX1 = cx - cropW / 2
    const winY1 = cy - cropH / 2
    const winX2 = winX1 + cropW
    const winY2 = winY1 + cropH

    const srcX1 = Math.max(0, Math.floor(winX1))
    const srcY1 = Math.max(0, Math.floor(winY1))
    const srcX2 = Math.min(srcWidth, Math.ceil(winX2))
    const srcY2 = Math.min(srcHeight, Math.ceil(winY2))
    const dstX1 = Math.max(0, Math.floor((srcX1 - winX1) * s))
    const dstY1 = Math.max(0, Math.floor((srcY1 - winY1) * s))
    const dstX2 = Math.min(outputWidth, Math.ceil((srcX2 - winX1) * s))
    const dstY2 = Math.min(outputHeight, Math.ceil((srcY2 - winY1) * s))

    const srcWW = clamp(srcX2 - srcX1, 0, srcWidth)
    const srcHH = clamp(srcY2 - srcY1, 0, srcHeight)
    const dstWW = clamp(dstX2 - dstX1, 0, outputWidth)
    const dstHH = clamp(dstY2 - dstY1, 0, outputHeight)

    if (srcWW > 0 && srcHH > 0 && dstWW > 0 && dstHH > 0) {
        ctx.imageSmoothingEnabled = true
        ctx.imageSmoothingQuality = 'high'
        ctx.drawImage(srcCanvas, srcX1, srcY1, srcWW, srcHH, dstX1, dstY1, dstWW, dstHH)
    }
}

type DetailedCropParams = {
    s: number
    winX1: number
    winY1: number
    dstX1: number
    dstY1: number
    dstX2: number
    dstY2: number
}

function getDetailedCropParams(
    outputWidth: number,
    outputHeight: number,
    srcWidth: number,
    srcHeight: number,
    det: TimedBBox,
    zoomMul: number = 1
): DetailedCropParams {
    const [x1p, y1p, x2p, y2p] = det.bbox
    const x1 = x1p * srcWidth
    const y1 = y1p * srcHeight
    const x2 = x2p * srcWidth
    const y2 = y2p * srcHeight
    const bboxW = Math.max(1, x2 - x1)
    const bboxH = Math.max(1, y2 - y1)

    const sHeight = (TARGET_BBOX_HEIGHT_RATIO * outputHeight) / bboxH
    const sWidthLimit = outputWidth / bboxW
    const sBase = clamp(Math.min(sHeight, sWidthLimit), MIN_SCALE, MAX_SCALE)
    const s = clamp(sBase * Math.max(1e-6, zoomMul), MIN_SCALE, MAX_SCALE)

    const cx = (x1 + x2) * 0.5
    const cy = (y1 + y2) * 0.5
    const cropW = outputWidth / s
    const cropH = outputHeight / s
    const winX1 = cx - cropW / 2
    const winY1 = cy - cropH / 2
    const winX2 = winX1 + cropW
    const winY2 = winY1 + cropH

    const srcX1 = Math.max(0, Math.floor(winX1))
    const srcY1 = Math.max(0, Math.floor(winY1))
    const srcX2 = Math.min(srcWidth, Math.ceil(winX2))
    const srcY2 = Math.min(srcHeight, Math.ceil(winY2))
    const dstX1 = Math.max(0, Math.floor((srcX1 - winX1) * s))
    const dstY1 = Math.max(0, Math.floor((srcY1 - winY1) * s))
    const dstX2 = Math.min(outputWidth, Math.ceil((srcX2 - winX1) * s))
    const dstY2 = Math.min(outputHeight, Math.ceil((srcY2 - winY1) * s))

    return { s, winX1, winY1, dstX1, dstY1, dstX2, dstY2 }
}

function drawAnnotationDot(ctx: Ctx2D, x: number, y: number, size: number, color: string) {
    ctx.beginPath()
    ctx.fillStyle = color
    ctx.arc(x, y, Math.max(1, size * 0.5), 0, Math.PI * 2)
    ctx.fill()
}

function drawAnnotationsOverview(ctx: Ctx2D, strokes: AnnotationStroke[], vidW: number, vidH: number, scale: number) {
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    for (const stroke of strokes) {
        if (stroke.points.length === 0) continue
        const lineWidth = Math.max(1, stroke.width / scale)
        if (stroke.points.length === 1) {
            const p = stroke.points[0]
            const x = p.x * vidW - vidW * 0.5
            const y = p.y * vidH - vidH * 0.5
            drawAnnotationDot(ctx, x, y, lineWidth, stroke.color)
            continue
        }
        ctx.strokeStyle = stroke.color
        ctx.lineWidth = lineWidth
        ctx.beginPath()
        stroke.points.forEach((p, idx) => {
            const x = p.x * vidW - vidW * 0.5
            const y = p.y * vidH - vidH * 0.5
            if (idx === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
        })
        ctx.stroke()
    }
}

function drawAnnotationsDetailed(
    ctx: Ctx2D,
    strokes: AnnotationStroke[],
    outputWidth: number,
    outputHeight: number,
    vidW: number,
    vidH: number,
    det: TimedBBox,
    zoomMul: number = 1
) {
    const params = getDetailedCropParams(outputWidth, outputHeight, vidW, vidH, det, zoomMul)
    const dstW = params.dstX2 - params.dstX1
    const dstH = params.dstY2 - params.dstY1
    if (dstW <= 0 || dstH <= 0) return

    ctx.save()
    ctx.beginPath()
    ctx.rect(params.dstX1, params.dstY1, dstW, dstH)
    ctx.clip()
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    for (const stroke of strokes) {
        if (stroke.points.length === 0) continue
        const lineWidth = Math.max(1, stroke.width)
        if (stroke.points.length === 1) {
            const p = stroke.points[0]
            const x = (p.x * vidW - params.winX1) * params.s
            const y = (p.y * vidH - params.winY1) * params.s
            drawAnnotationDot(ctx, x, y, lineWidth, stroke.color)
            continue
        }
        ctx.strokeStyle = stroke.color
        ctx.lineWidth = lineWidth
        ctx.beginPath()
        stroke.points.forEach((p, idx) => {
            const x = (p.x * vidW - params.winX1) * params.s
            const y = (p.y * vidH - params.winY1) * params.s
            if (idx === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
        })
        ctx.stroke()
    }
    ctx.restore()
}

function drawStabilizationTransforms(
    ctx: CanvasRenderingContext2D,
    player: PlayerState,
    nowFrameIndex: number,
    sBase: number,
    cx: number,
    cy: number
) {
    // Debug: draw stabilization trail (last ~30 samples) anchored at center (relative to current)
    try {
        const N = 30
        const sScale = sBase
        const pts: Array<{ x: number; y: number }> = []
        const siNow = player.getStabilizationAtFrame(nowFrameIndex)
        const vx0 = sScale * siNow.dx
        const vy0 = sScale * siNow.dy
        for (let i = 0; i < N; i++) {
            const tFrame = Math.max(0, nowFrameIndex - i)
            const si = player.getStabilizationAtFrame(tFrame)
            const vx = sScale * si.dx
            const vy = sScale * si.dy
            const px = cx + (vx - vx0)
            const py = cy + (vy - vy0)
            pts.push({ x: px, y: py })
        }
        if (pts.length >= 2) {
            ctx.save()
            ctx.lineWidth = 2
            for (let i = 0; i < pts.length - 1; i++) {
                const a = Math.max(0.15, 1 - i / pts.length)
                ctx.strokeStyle = `rgba(56,189,248,${a})` // cyan-400 with fade
                ctx.beginPath()
                ctx.moveTo(pts[i].x, pts[i].y)
                ctx.lineTo(pts[i + 1].x, pts[i + 1].y)
                ctx.stroke()
            }
            // head marker
            ctx.fillStyle = 'rgba(56,189,248,0.9)'
            ctx.beginPath()
            ctx.arc(cx, cy, 3, 0, Math.PI * 2)
            ctx.fill()
            ctx.restore()
        }
    } catch {}
}

export function drawFrame(
    canvas: HTMLCanvasElement,
    containerEl: HTMLElement,
    source: CanvasImageSource,
    sourceSize: { width: number; height: number },
    player: PlayerState,
    ov: OverviewView,
    annotations: AnnotationStroke[] = [],
    frameIndex: number,
    dominantOrientationDeg: number = 0
) {
    const rect = containerEl.getBoundingClientRect()
    const cssW = Math.max(1, Math.floor(rect.width))
    const cssH = Math.max(1, Math.floor(rect.height))
    const ctx = ensureCanvasSize(canvas, cssW, cssH)

    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, cssW, cssH)

    // Prepare source draw surface with orientation applied (reuse a single offscreen canvas)
    const offscreen = getSharedOffscreenCanvas()
    const rotatedVideo = drawRotatedToCanvas(source, offscreen, dominantOrientationDeg, sourceSize)

    // Current frame for stabilization/detection lookup
    const nowFrame = frameIndex
    const sourceCanvas: HTMLCanvasElement = offscreen

    if (player.mode === 'overview') {
        const base = computeBaseRect(cssW, cssH, rotatedVideo.width, rotatedVideo.height)
        const z = ov.zoom
        const cx = base.x + base.w * 0.5 + ov.offsetX
        const cy = base.y + base.h * 0.5 + ov.offsetY
        const sBase = (base.w / rotatedVideo.width) * z
        const stab = player.getStabilizationAtFrame(nowFrame)

        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(sBase, sBase)
        // Apply cumulative stabilization as pre-apply transform: translate then rotate
        ctx.translate(stab.dx, stab.dy)
        ctx.rotate(stab.da)
        ctx.imageSmoothingEnabled = true
        ctx.imageSmoothingQuality = 'high'
        ctx.drawImage(offscreen, -rotatedVideo.width * 0.5, -rotatedVideo.height * 0.5)

        if (true) {
            // TODO false
            drawStabilizationTransforms(ctx, player, nowFrame, sBase, cx, cy)
        }

        // Draw detections under same transform
        for (const t of player.tracks) {
            const isHovered = ov.hoveredTrackId === t.track_id
            // TODO if (!isHovered) continue

            const det = player.getDetectionAtFrame(t.track_id, nowFrame)
            if (!det) continue
            const [x1p, y1p, x2p, y2p] = det.bbox
            const x1 = x1p * rotatedVideo.width - rotatedVideo.width * 0.5
            const y1 = y1p * rotatedVideo.height - rotatedVideo.height * 0.5
            const w = Math.max(1, (x2p - x1p) * rotatedVideo.width)
            const h = Math.max(1, (y2p - y1p) * rotatedVideo.height)

            ctx.strokeStyle = '#10b981'
            ctx.lineWidth = 2 / sBase
            ctx.strokeRect(Math.round(x1) + 0.5, Math.round(y1) + 0.5, Math.round(w), Math.round(h))
        }
        if (annotations.length > 0) {
            drawAnnotationsOverview(ctx, annotations, rotatedVideo.width, rotatedVideo.height, sBase)
        }
        ctx.restore()
    } else if (player.mode === 'detailed' && player.currentTrackId != null) {
        const det = player.getClosestDetectionAtFrame(player.currentTrackId, nowFrame)

        const vidW = rotatedVideo.width
        const vidH = rotatedVideo.height
        // Reuse shared crop-draw logic (same as export path).
        const zMul = ov.detailedZoom ?? 1
        drawDetailedCrop(ctx, cssW, cssH, sourceCanvas, vidW, vidH, det, zMul)
        if (annotations.length > 0) {
            drawAnnotationsDetailed(ctx, annotations, cssW, cssH, vidW, vidH, det, zMul)
        }
    }
}

export function screenPointToVideoNorm(
    px: number,
    py: number,
    outW: number,
    outH: number,
    player: PlayerState,
    videoWidth: number,
    videoHeight: number,
    frameIndex: number,
    ov: OverviewView,
    dominantOrientationDeg: number = 0
): AnnotationPoint | null {
    const { width, height } = getRotatedDimensions(videoWidth, videoHeight, dominantOrientationDeg)
    if (player.mode === 'overview') {
        const base = computeBaseRect(outW, outH, width, height)
        const stab = player.getStabilizationAtFrame(frameIndex)

        const z = ov.zoom
        const cx = base.x + base.w * 0.5 + ov.offsetX
        const cy = base.y + base.h * 0.5 + ov.offsetY
        const sBase = (base.w / width) * z

        const dx0 = px - cx
        const dy0 = py - cy
        const dx1 = dx0 / sBase
        const dy1 = dy0 / sBase
        const dx2 = dx1 - stab.dx
        const dy2 = dy1 - stab.dy

        const cos = Math.cos(-stab.da)
        const sin = Math.sin(-stab.da)
        const dx3 = dx2 * cos - dy2 * sin
        const dy3 = dx2 * sin + dy2 * cos

        const xNorm = (dx3 + width * 0.5) / width
        const yNorm = (dy3 + height * 0.5) / height
        if (xNorm < 0 || xNorm > 1 || yNorm < 0 || yNorm > 1) return null
        return { x: xNorm, y: yNorm }
    }

    if (player.mode === 'detailed' && player.currentTrackId != null) {
        const det = player.getClosestDetectionAtFrame(player.currentTrackId, frameIndex)
        const params = getDetailedCropParams(outW, outH, width, height, det, ov.detailedZoom ?? 1)
        if (px < params.dstX1 || px > params.dstX2 || py < params.dstY1 || py > params.dstY2) return null
        const vx = params.winX1 + px / params.s
        const vy = params.winY1 + py / params.s
        const xNorm = vx / width
        const yNorm = vy / height
        if (xNorm < 0 || xNorm > 1 || yNorm < 0 || yNorm > 1) return null
        return { x: xNorm, y: yNorm }
    }

    return null
}

export function pickTrackAtScreenPoint(
    px: number,
    py: number,
    outW: number,
    outH: number,
    player: PlayerState,
    videoWidth: number,
    videoHeight: number,
    frameIndex: number,
    ov: OverviewView,
    dominantOrientationDeg: number = 0
): number | null {
    const { width, height } = getRotatedDimensions(videoWidth, videoHeight, dominantOrientationDeg)
    const base = computeBaseRect(outW, outH, width, height)
    const stab = player.getStabilizationAtFrame(frameIndex)

    const z = ov.zoom
    const cx = base.x + base.w * 0.5 + ov.offsetX
    const cy = base.y + base.h * 0.5 + ov.offsetY
    const sBase = (base.w / width) * z

    // Undo transform chain: Translate(cx,cy) -> Scale(s) -> Translate(stab) -> Rotate(stab)

    // 1. Undo Screen Center Translation
    const dx0 = px - cx
    const dy0 = py - cy

    // 2. Undo Scale
    const dx1 = dx0 / sBase
    const dy1 = dy0 / sBase

    // 3. Undo Stabilization Translation
    const dx2 = dx1 - stab.dx
    const dy2 = dy1 - stab.dy

    // 4. Undo Stabilization Rotation
    const cos = Math.cos(-stab.da)
    const sin = Math.sin(-stab.da)
    const dx3 = dx2 * cos - dy2 * sin
    const dy3 = dx2 * sin + dy2 * cos

    // Convert from centered coords to normalized [0,1]
    const xNorm = (dx3 + width * 0.5) / width
    const yNorm = (dy3 + height * 0.5) / height

    for (const t of player.tracks) {
        const det = player.getDetectionAtFrame(t.track_id, frameIndex)
        if (!det) continue
        const [x1p, y1p, x2p, y2p] = det.bbox

        if (xNorm >= x1p && xNorm <= x2p && yNorm >= y1p && yNorm <= y2p) {
            return t.track_id
        }
    }
    return null
}
