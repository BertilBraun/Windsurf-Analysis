/**
 * @file Hook for managing playback speed state.
 */

import React from 'react'
import { clamp } from '../utils/clamp'

export const PLAYBACK_SPEED_RATES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] as const

/**
 * Manages playback speed state, cycling through predefined rates (0.25x to 8.0x).
 *
 * @param initial - The initial playback speed. Defaults to 1.0.
 * @returns An object containing the current speed and a function to increment or decrement it.
 */
export function usePlaybackSpeed(initial: number = 1.0) {
    const [speed, setSpeed] = React.useState(initial)

    const bumpSpeed = React.useCallback((down: boolean) => {
        setSpeed(prev => {
            const idx = clamp(
                PLAYBACK_SPEED_RATES.indexOf(prev as any) + (down ? -1 : 1),
                0,
                PLAYBACK_SPEED_RATES.length - 1
            )
            return PLAYBACK_SPEED_RATES[idx] ?? 1.0
        })
    }, [])

    const setSpeedSafe = React.useCallback((next: number) => {
        if (!Number.isFinite(next)) return
        const nearest =
            PLAYBACK_SPEED_RATES.find(rate => rate === next) ??
            PLAYBACK_SPEED_RATES.reduce(
                (best, rate) => (Math.abs(rate - next) < Math.abs(best - next) ? rate : best),
                1.0
            )
        setSpeed(nearest)
    }, [])

    return { speed, bumpSpeed, setSpeed: setSpeedSafe, rates: [...PLAYBACK_SPEED_RATES] as number[] }
}
