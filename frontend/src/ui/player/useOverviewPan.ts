import React from 'react'
import { clamp } from '../utils/clamp'
import type { ZoomOffset } from '../hooks/useZoom'
import { computeBaseRect } from './renderMath'
import { getRotatedDimensions } from './rotation'

type PointerHandlers = {
    onPointerDown?: (e: React.PointerEvent<HTMLCanvasElement>) => void
    onPointerMove?: (e: React.PointerEvent<HTMLCanvasElement>) => void
    onPointerUp?: (e: React.PointerEvent<HTMLCanvasElement>) => void
    onPointerCancel?: (e: React.PointerEvent<HTMLCanvasElement>) => void
}

export function useOverviewPan(
    params: {
        enabled: boolean
        zoom: number
        offset: ZoomOffset
        setOffset: React.Dispatch<React.SetStateAction<ZoomOffset>>
        containerRef: React.RefObject<HTMLElement | null>
        videoSize: { width: number; height: number }
        dominantOrientationDeg: number
        onPanStart?: () => void
    },
    delegate?: PointerHandlers
): {
    canPan: boolean
    isPanning: boolean
    shouldSuppressClick: () => boolean
    onPointerDown: (e: React.PointerEvent<HTMLCanvasElement>) => void
    onPointerMove: (e: React.PointerEvent<HTMLCanvasElement>) => void
    onPointerUp: (e: React.PointerEvent<HTMLCanvasElement>) => void
    onPointerCancel: (e: React.PointerEvent<HTMLCanvasElement>) => void
} {
    const { enabled, zoom, offset, setOffset, containerRef, videoSize, dominantOrientationDeg, onPanStart } = params

    const [isPanning, setIsPanning] = React.useState(false)
    const panStateRef = React.useRef<{
        pointerId: number
        startClientX: number
        startClientY: number
        startOffsetX: number
        startOffsetY: number
        started: boolean
    } | null>(null)
    const suppressClickRef = React.useRef(false)

    const canPan = enabled && zoom > 1.0001 && videoSize.width > 0 && videoSize.height > 0

    const shouldSuppressClick = React.useCallback(() => suppressClickRef.current, [])

    const onPointerDown = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            delegate?.onPointerDown?.(e)

            if (!canPan) return
            if (e.button !== 0) return

            panStateRef.current = {
                pointerId: e.pointerId,
                startClientX: e.clientX,
                startClientY: e.clientY,
                startOffsetX: offset.x,
                startOffsetY: offset.y,
                started: false,
            }
            e.currentTarget.setPointerCapture(e.pointerId)
        },
        [canPan, delegate, offset.x, offset.y]
    )

    const onPointerMove = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            delegate?.onPointerMove?.(e)

            const st = panStateRef.current
            if (!st) return
            if (st.pointerId !== e.pointerId) return

            const dx = e.clientX - st.startClientX
            const dy = e.clientY - st.startClientY

            if (!st.started) {
                const thresholdPx = 4
                if (dx * dx + dy * dy < thresholdPx * thresholdPx) return
                st.started = true
                setIsPanning(true)
                onPanStart?.()
            }

            const container = containerRef.current
            if (!container) return

            const rect = container.getBoundingClientRect()
            const outW = Math.max(1, rect.width)
            const outH = Math.max(1, rect.height)
            const { width: vidW, height: vidH } = getRotatedDimensions(
                videoSize.width,
                videoSize.height,
                dominantOrientationDeg
            )
            const base = computeBaseRect(outW, outH, Math.max(1, vidW), Math.max(1, vidH))
            const imageW = base.w * zoom
            const imageH = base.h * zoom
            const maxOffsetX = Math.max(0, (imageW - outW) / 2)
            const maxOffsetY = Math.max(0, (imageH - outH) / 2)

            e.preventDefault()
            setOffset({
                x: clamp(st.startOffsetX + dx, -maxOffsetX, maxOffsetX),
                y: clamp(st.startOffsetY + dy, -maxOffsetY, maxOffsetY),
            })
        },
        [containerRef, delegate, dominantOrientationDeg, onPanStart, setOffset, videoSize.height, videoSize.width, zoom]
    )

    const endPan = React.useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
        const st = panStateRef.current
        if (!st) return
        if (st.pointerId !== e.pointerId) return

        panStateRef.current = null
        if (e.currentTarget.hasPointerCapture(e.pointerId)) {
            e.currentTarget.releasePointerCapture(e.pointerId)
        }

        if (st.started) {
            suppressClickRef.current = true
            requestAnimationFrame(() => {
                suppressClickRef.current = false
            })
        }
        setIsPanning(false)
    }, [])

    const onPointerUp = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            delegate?.onPointerUp?.(e)
            endPan(e)
        },
        [delegate, endPan]
    )

    const onPointerCancel = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            delegate?.onPointerCancel?.(e)
            endPan(e)
        },
        [delegate, endPan]
    )

    return { canPan, isPanning, shouldSuppressClick, onPointerDown, onPointerMove, onPointerUp, onPointerCancel }
}

