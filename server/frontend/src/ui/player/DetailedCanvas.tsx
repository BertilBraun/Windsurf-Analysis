import React from 'react'
import { PlayerState, findNearestDetectionByTime } from './state'

const TARGET_BBOX_HEIGHT_RATIO = 0.7
const MIN_SCALE = 0.2
const MAX_SCALE = 10.0

export const DetailedCanvas: React.FC<{
    state: PlayerState
    videoRef: React.RefObject<HTMLVideoElement>
}> = ({ state, videoRef }) => {
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null)

    React.useEffect(() => {
        let raf = 0
        const v = videoRef.current
        const c = canvasRef.current
        if (!v || !c) return

        const ctx = c.getContext('2d')
        if (!ctx) return

        const draw = () => {
            const trackId = state.currentTrackId
            if (state.mode !== 'detailed' || trackId == null) {
                // hide canvas if not detailed
                c.style.display = 'none'
                raf = requestAnimationFrame(draw)
                return
            }
            c.style.display = 'block'

            // canvas sizing to match displayed video size (CSS pixels), with HiDPI support
            const outW = v.clientWidth || 1
            const outH = v.clientHeight || 1
            const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1))
            if (c.width !== Math.max(1, Math.floor(outW * dpr)) || c.height !== Math.max(1, Math.floor(outH * dpr))) {
                c.width = Math.max(1, Math.floor(outW * dpr))
                c.height = Math.max(1, Math.floor(outH * dpr))
            }
            c.style.width = `${outW}px`
            c.style.height = `${outH}px`
            ctx.setTransform(1, 0, 0, 1, 0, 0)
            ctx.scale(dpr, dpr)
            ctx.fillStyle = '#000'
            ctx.fillRect(0, 0, outW, outH)

            const vidW = v.videoWidth || 1
            const vidH = v.videoHeight || 1

            // nearest detection around current time
            const arr = state.detectionTimesByTrack.get(trackId) || []
            const nearest = findNearestDetectionByTime(arr, state.currentTimeSec, 0.2)
            if (!nearest) {
                // fallback: draw full frame letterboxed into canvas
                const scale = Math.min(outW / vidW, outH / vidH)
                const dw = Math.floor(vidW * scale)
                const dh = Math.floor(vidH * scale)
                const dx = Math.floor((outW - dw) / 2)
                const dy = Math.floor((outH - dh) / 2)
                try {
                    ctx.drawImage(v, 0, 0, vidW, vidH, dx, dy, dw, dh)
                } catch {}
                raf = requestAnimationFrame(draw)
                return
            }

            const [x1p, y1p, x2p, y2p] = nearest.detection.bbox
            const x1 = x1p * vidW
            const y1 = y1p * vidH
            const x2 = x2p * vidW
            const y2 = y2p * vidH
            const bboxW = Math.max(1, x2 - x1)
            const bboxH = Math.max(1, y2 - y1)

            // Choose scale to respect target bbox height and ensure bbox width fits
            const sHeight = (TARGET_BBOX_HEIGHT_RATIO * outH) / bboxH
            const sWidthLimit = outW / bboxW
            const s = Math.max(MIN_SCALE, Math.min(MAX_SCALE, Math.min(sHeight, sWidthLimit)))

            // Crop window in source image coords centered on bbox
            const cx = (x1 + x2) * 0.5
            const cy = (y1 + y2) * 0.5
            const cropW = outW / s
            const cropH = outH / s
            const winX1 = cx - cropW / 2
            const winY1 = cy - cropH / 2
            const winX2 = winX1 + cropW
            const winY2 = winY1 + cropH

            // Clamp to video bounds and compute destination placement
            const srcX1 = Math.max(0, Math.floor(winX1))
            const srcY1 = Math.max(0, Math.floor(winY1))
            const srcX2 = Math.min(vidW, Math.ceil(winX2))
            const srcY2 = Math.min(vidH, Math.ceil(winY2))
            const dstX1 = Math.max(0, Math.floor((srcX1 - winX1) * s))
            const dstY1 = Math.max(0, Math.floor((srcY1 - winY1) * s))
            const dstX2 = Math.min(outW, Math.ceil((srcX2 - winX1) * s))
            const dstY2 = Math.min(outH, Math.ceil((srcY2 - winY1) * s))
            const srcW = Math.max(0, srcX2 - srcX1)
            const srcH = Math.max(0, srcY2 - srcY1)
            const dstW = Math.max(0, dstX2 - dstX1)
            const dstH = Math.max(0, dstY2 - dstY1)

            if (srcW > 0 && srcH > 0 && dstW > 0 && dstH > 0) {
                try {
                    ctx.imageSmoothingEnabled = s < 1 ? true : true
                    ctx.imageSmoothingQuality = 'high'
                    ctx.drawImage(v, srcX1, srcY1, srcW, srcH, dstX1, dstY1, dstW, dstH)
                } catch {}
            }

            raf = requestAnimationFrame(draw)
        }

        raf = requestAnimationFrame(draw)
        return () => cancelAnimationFrame(raf)
    }, [state.mode, state.currentTrackId, state.currentTimeSec, videoRef.current])

    return <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, zIndex: 2, pointerEvents: 'none' }} />
}
