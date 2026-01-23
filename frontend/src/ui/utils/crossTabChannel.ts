type Envelope<T> = { at: number; payload: T }

function safeJsonParse<T>(raw: string): T | null {
    try {
        return JSON.parse(raw) as T
    } catch {
        return null
    }
}

export function createCrossTabChannel<T>(opts: { broadcastChannelName: string; storageKey: string }) {
    const { broadcastChannelName, storageKey } = opts

    const publish = (payload: T) => {
        const env: Envelope<T> = { at: Date.now(), payload }

        try {
            const bc = new BroadcastChannel(broadcastChannelName)
            bc.postMessage(env)
            bc.close()
        } catch {}

        try {
            localStorage.setItem(storageKey, JSON.stringify(env))
        } catch {}
    }

    const subscribe = (onPayload: (payload: T) => void) => {
        let bc: BroadcastChannel | null = null

        try {
            bc = new BroadcastChannel(broadcastChannelName)
            bc.onmessage = event => {
                const env = event?.data as Envelope<T> | undefined
                if (!env) return
                onPayload(env.payload)
            }
        } catch {
            bc = null
        }

        const onStorage = (e: StorageEvent) => {
            if (e.key !== storageKey) return
            const parsed = safeJsonParse<Envelope<T>>(String(e.newValue || ''))
            if (!parsed) return
            onPayload(parsed.payload)
        }
        window.addEventListener('storage', onStorage)

        return () => {
            window.removeEventListener('storage', onStorage)
            try {
                bc?.close()
            } catch {}
            bc = null
        }
    }

    return { publish, subscribe }
}

export function createCrossTabSignal(opts: { broadcastChannelName: string; storageKey: string }) {
    const chan = createCrossTabChannel<{ at: number }>(opts)
    return {
        notify: () => chan.publish({ at: Date.now() }),
        subscribe: (onNotify: () => void) => chan.subscribe(() => onNotify()),
    }
}

