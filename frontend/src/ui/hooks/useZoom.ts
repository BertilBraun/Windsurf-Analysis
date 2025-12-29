import React from 'react'

export type ZoomOffset = { x: number; y: number }

export function useZoom(
    containerRef: React.RefObject<HTMLDivElement>,
    opts?: {
        minZoom?: number
        maxZoom?: number
        step?: number
    }
) {
    const { minZoom = 1, maxZoom = Number.POSITIVE_INFINITY, step = 0.1 } = opts ?? {}
    const [zoom, setZoom] = React.useState(1)
    const [offset, setOffset] = React.useState<ZoomOffset>({ x: 0, y: 0 })

    const reset = React.useCallback(() => {
        setZoom(1)
        setOffset({ x: 0, y: 0 })
    }, [])

    const onWheelZoom = React.useCallback(
        (absX: number, absY: number, deltaY: number) => {
            const rect = containerRef.current?.getBoundingClientRect()
            const cx = rect ? absX - rect.left : absX
            const cy = rect ? absY - rect.top : absY
            const factor = 1 + (deltaY < 0 ? step : -step)
            const unclampedZoom = zoom * factor
            const nextZoom = Math.max(minZoom, Math.min(maxZoom, unclampedZoom))
            if (Math.abs(nextZoom - zoom) < 1e-9) return

            const scaleChange = nextZoom / Math.max(1e-9, zoom)
            const nextX = cx - scaleChange * (cx - offset.x)
            const nextY = cy - scaleChange * (cy - offset.y)
            setZoom(nextZoom)
            setOffset({ x: nextX, y: nextY })
        },
        [containerRef, zoom, offset, minZoom, maxZoom, step]
    )

    return { zoom, offset, onWheelZoom, reset }
}
