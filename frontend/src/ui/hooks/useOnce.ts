import React from 'react'
import { loadSetting, saveSetting } from '../utils/idb'

export type UseOnceResult = {
    used: boolean
    ready: boolean
    mark: () => void
}

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
