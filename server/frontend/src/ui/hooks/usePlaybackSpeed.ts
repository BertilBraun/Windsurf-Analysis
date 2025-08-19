import React from 'react'
import { clamp } from '../utils/clamp'

const RATES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

export function usePlaybackSpeed(initial: number = 1.0) {
    const [speed, setSpeed] = React.useState(initial)

    const bumpSpeed = React.useCallback((down: boolean) => {
        setSpeed(prev => {
            const idx = clamp(RATES.indexOf(prev) + (down ? -1 : 1), 0, RATES.length - 1)
            return RATES[idx]
        })
    }, [])

    return { speed, bumpSpeed }
}
