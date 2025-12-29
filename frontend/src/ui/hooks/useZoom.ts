import React from 'react'
import { useCappedValue } from './useCappedValue'
import { clamp } from '../utils/clamp'

export type ZoomOffset = { x: number; y: number }

export function useZoom(opts?: { minZoom?: number; maxZoom?: number; step?: number }) {
    const { minZoom = 1, maxZoom = Number.POSITIVE_INFINITY, step = 0.1 } = opts ?? {}
    const { value: zoom, set: setZoom, reset: resetZoom } = useCappedValue(1, minZoom, maxZoom)
    const [offset, setOffset] = React.useState<ZoomOffset>({ x: 0, y: 0 })

    const reset = React.useCallback(() => {
        setOffset({ x: 0, y: 0 })
        resetZoom()
    }, [resetZoom])

    const onWheelZoom = React.useCallback(
        (centeredX: number, centeredY: number, deltaY: number) => {
            const factor = 1 + (deltaY < 0 ? step : -step)
            const prevZoom = zoom
            const nextZoom = clamp(prevZoom * factor, minZoom, maxZoom)
            if (Math.abs(nextZoom - prevZoom) < 1e-9) return

            const scaleChange = nextZoom / prevZoom
            const nextX = centeredX - scaleChange * (centeredX - offset.x)
            const nextY = centeredY - scaleChange * (centeredY - offset.y)

            setZoom(nextZoom)
            setOffset({ x: nextX, y: nextY })
        },
        [step, zoom, minZoom, maxZoom, offset.x, offset.y, setZoom]
    )

    return { zoom, offset, onWheelZoom, reset }
}
