import React from 'react'

export type ZoomOffset = { x: number; y: number }

export function useZoom(containerRef: React.RefObject<HTMLDivElement>) {
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
            let nextZoom = zoom * (1 + (deltaY < 0 ? 0.1 : -0.1))
            if (nextZoom <= 1) {
                reset()
                return
            }
            const scaleChange = nextZoom / zoom
            const nextX = cx - scaleChange * (cx - offset.x)
            const nextY = cy - scaleChange * (cy - offset.y)
            setZoom(nextZoom)
            setOffset({ x: nextX, y: nextY })
        },
        [containerRef, zoom, offset, reset]
    )

    return { zoom, offset, onWheelZoom, reset }
}
