import React from 'react'
import { DrawOverlay, DRAW_COLOR_OPTIONS, DRAW_WIDTH_OPTIONS, type DrawTool } from './DrawOverlay'
import type { AnnotationPoint, AnnotationStroke } from './rendering'

type GetDrawPoint = (e: React.PointerEvent<HTMLCanvasElement>) => AnnotationPoint | null

export function useAnnotations(
    getDrawPoint: GetDrawPoint,
    params?: {
        drawMode?: boolean
        isExporting?: boolean
        isPlaying?: boolean
        currentFrameIndex?: number
    }
): {
    getVisibleAnnotations: (frameIndex: number) => AnnotationStroke[]
    undo: () => void
    reset: () => void
    drawModal: React.ReactNode
    onPointerDown: (e: React.PointerEvent<HTMLCanvasElement>) => void
    onPointerMove: (e: React.PointerEvent<HTMLCanvasElement>) => void
    onPointerUp: (e: React.PointerEvent<HTMLCanvasElement>) => void
    onPointerCancel: (e: React.PointerEvent<HTMLCanvasElement>) => void
} {
    const { drawMode = false, isExporting = false, isPlaying = false, currentFrameIndex = 0 } = params ?? {}

    const [drawTool, setDrawTool] = React.useState<DrawTool>('line')
    const [drawColor, setDrawColor] = React.useState<string>(DRAW_COLOR_OPTIONS[0])
    const [drawWidth, setDrawWidth] = React.useState<number>(DRAW_WIDTH_OPTIONS[2] ?? 5)

    const [annotations, setAnnotations] = React.useState<AnnotationStroke[]>([])
    const [activeStroke, setActiveStroke] = React.useState<AnnotationStroke | null>(null)

    const activeStrokeRef = React.useRef<AnnotationStroke | null>(null)
    const activePointerIdRef = React.useRef<number | null>(null)

    const reset = React.useCallback(() => {
        setAnnotations([])
        setActiveStroke(null)
        activeStrokeRef.current = null
        activePointerIdRef.current = null
        setDrawTool('line')
        setDrawColor(DRAW_COLOR_OPTIONS[0])
        setDrawWidth(DRAW_WIDTH_OPTIONS[2] ?? 5)
    }, [])

    const endStroke = React.useCallback((el: HTMLCanvasElement, pointerId: number) => {
        activePointerIdRef.current = null
        activeStrokeRef.current = null
        setActiveStroke(null)
        if (el.hasPointerCapture(pointerId)) el.releasePointerCapture(pointerId)
    }, [])

    const getVisibleAnnotations = React.useCallback(
        (frameIndex: number) => {
            const visible = annotations.filter(stroke => stroke.frameIndex === frameIndex)
            if (activeStroke && activeStroke.frameIndex === frameIndex) visible.push(activeStroke)
            return visible
        },
        [annotations, activeStroke]
    )

    const undo = React.useCallback(() => {
        if (activeStrokeRef.current) {
            activeStrokeRef.current = null
            activePointerIdRef.current = null
            setActiveStroke(null)
            return
        }
        setAnnotations(prev => (prev.length ? prev.slice(0, -1) : prev))
    }, [])

    const hasVisibleAnnotations = React.useMemo(() => {
        return getVisibleAnnotations(currentFrameIndex).length > 0
    }, [getVisibleAnnotations, currentFrameIndex])

    const onClearAnnotations = React.useCallback(() => {
        const now = currentFrameIndex
        setAnnotations(prev => prev.filter(stroke => stroke.frameIndex !== now))
        if (activeStrokeRef.current?.frameIndex === now) {
            activeStrokeRef.current = null
            activePointerIdRef.current = null
            setActiveStroke(null)
        }
    }, [currentFrameIndex])

    const toolEnabled = drawMode && !isExporting && !isPlaying

    const startStroke = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            const point = getDrawPoint(e)
            if (!point) return

            const stroke: AnnotationStroke = {
                id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
                frameIndex: currentFrameIndex,
                color: drawColor,
                width: drawWidth,
                points: drawTool === 'line' ? [point, point] : [point],
            }

            activePointerIdRef.current = e.pointerId
            activeStrokeRef.current = stroke
            e.currentTarget.setPointerCapture(e.pointerId)
            setActiveStroke(stroke)
        },
        [getDrawPoint, currentFrameIndex, drawColor, drawWidth, drawTool]
    )

    const onPointerDown = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            if (!toolEnabled) return
            if (e.button !== 0) return
            e.preventDefault()

            if (activeStrokeRef.current) {
                endStroke(e.currentTarget, e.pointerId)
            }
            startStroke(e)
        },
        [toolEnabled, startStroke, endStroke]
    )

    const onPointerMove = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            if (!toolEnabled) return
            const stroke = activeStrokeRef.current
            if (!stroke) return
            if (activePointerIdRef.current !== e.pointerId) return

            const point = getDrawPoint(e)
            if (!point) return

            if (drawTool === 'line') {
                const next: AnnotationStroke = { ...stroke, points: [stroke.points[0], point] }
                activeStrokeRef.current = next
                setActiveStroke(next)
                return
            }

            const last = stroke.points[stroke.points.length - 1]
            const dx = point.x - last.x
            const dy = point.y - last.y
            if (dx * dx + dy * dy < 1e-7) return
            const next: AnnotationStroke = { ...stroke, points: [...stroke.points, point] }
            activeStrokeRef.current = next
            setActiveStroke(next)
        },
        [toolEnabled, getDrawPoint, drawTool]
    )

    const finalizeActiveStroke = React.useCallback(
        (el: HTMLCanvasElement, pointerId: number) => {
            const stroke = activeStrokeRef.current
            if (stroke) setAnnotations(prev => [...prev, stroke])
            endStroke(el, pointerId)
        },
        [endStroke]
    )

    const onPointerUp = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            if (!toolEnabled) return
            if (activePointerIdRef.current !== e.pointerId) return
            finalizeActiveStroke(e.currentTarget, e.pointerId)
        },
        [toolEnabled, finalizeActiveStroke]
    )

    const onPointerCancel = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            if (!toolEnabled) return
            if (activePointerIdRef.current !== e.pointerId) return
            endStroke(e.currentTarget, e.pointerId)
        },
        [toolEnabled, endStroke]
    )

    const drawModal = drawMode ? (
        <DrawOverlay
            drawTool={drawTool}
            onDrawToolChange={setDrawTool}
            drawWidth={drawWidth}
            onDrawWidthChange={setDrawWidth}
            drawColor={drawColor}
            onDrawColorChange={setDrawColor}
            onClearAnnotations={onClearAnnotations}
            hasVisibleAnnotations={hasVisibleAnnotations}
        />
    ) : null

    return {
        getVisibleAnnotations,
        undo,
        reset,
        drawModal,
        onPointerDown,
        onPointerMove,
        onPointerUp,
        onPointerCancel,
    }
}

