import React from 'react'
import { clamp } from '../utils/clamp'

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

