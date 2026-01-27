/**
 * @fileoverview Hook for tracking one-time persistent actions.
 */

import React from 'react'
import { loadSetting, saveSetting } from '../utils/idb'

/**
 * Result of the {@link useOnce} hook.
 */
export type UseOnceResult = {
    /** True if the action has been marked as used. */
    used: boolean
    /** True if the initial state has been loaded from storage. */
    ready: boolean
    /** Persistently marks the action as used. */
    mark: () => void
}

/**
 * Tracks whether a specific action has been performed once, using IndexedDB for persistence.
 *
 * @param key - Unique identifier for the action.
 * @returns The current state and a function to mark the action as used.
 */
export function useOnce(key: string): UseOnceResult {
    const [state, setState] = React.useState<{ used: boolean; ready: boolean }>({ used: false, ready: false })

    React.useEffect(() => {
        setState({ used: false, ready: false })
        let cancelled = false
        loadSetting<boolean>(key).then(saved => {
            if (cancelled) return
            setState({ used: !!saved, ready: true })
        })
        return () => {
            cancelled = true
        }
    }, [key])

    const mark = React.useCallback(() => {
        setState({ used: true, ready: true })
        void saveSetting(key, true)
    }, [key])

    return { used: state.used, ready: state.ready, mark }
}
