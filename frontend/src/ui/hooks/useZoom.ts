/**
 * @module useZoom
 * Provides logic for managing zoom levels and coordinate offsets, typically for canvas or viewport interactions.
 */

import React from 'react'
import { useCappedValue } from './useCappedValue'
import { clamp } from '../utils/clamp'

/**
 * Represents a 2D coordinate offset used for panning during zoom operations.
 */
export type ZoomOffset = { x: number; y: number }

/**
 * A hook to manage zoom state and calculate offsets relative to a focal point.
 *
 * @param opts - Configuration for zoom constraints and step increments.
 * @param opts.minZoom - The minimum allowed zoom level. Defaults to 1.
 * @param opts.maxZoom - The maximum allowed zoom level. Defaults to Infinity.
 * @param opts.step - The zoom increment factor applied per wheel event. Defaults to 0.1.
 * @returns An object containing:
 * - `zoom`: The current zoom scale factor.
 * - `offset`: The current {x, y} translation offset.
 * - `setOffset`: Function to manually update the translation offset.
 * - `onWheelZoom`: A handler to update zoom and offset based on a wheel event at a specific coordinate.
 * - `reset`: A function to reset zoom to 1 and offset to {0, 0}.
 */
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

    return { zoom, offset, setOffset, onWheelZoom, reset }
}
