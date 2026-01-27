/**
 * Hook for managing a numeric value constrained within a specific range.
 */
import React from 'react'
import { clamp } from '../utils/clamp'

/**
 * Manages a numeric state value that is automatically clamped between min and max.
 *
 * @param initialValue - The initial value to set (will be clamped).
 * @param min - The lower bound.
 * @param max - The upper bound.
 * @returns An object containing the current value, a setter that enforces bounds, and a reset function.
 */
export function useCappedValue(initialValue: number, min: number, max: number) {
    const [value, setValue] = React.useState(() => clamp(initialValue, min, max))

    const setCapped = React.useCallback(
        (next: number | ((prev: number) => number)) => {
            setValue(prev => clamp(typeof next === 'function' ? next(prev) : next, min, max))
        },
        [min, max]
    )

    const reset = React.useCallback(() => {
        setValue(clamp(initialValue, min, max))
    }, [initialValue, min, max])

    return { value, set: setCapped, reset }
}
