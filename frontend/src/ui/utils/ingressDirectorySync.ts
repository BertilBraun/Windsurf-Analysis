const BC_NAME = 'windsurf:ingressDirectory'
const STORAGE_KEY = 'windsurf:ingressDirectory:changedAt'

export function notifyIngressDirectoryChanged() {
    try {
        const bc = new BroadcastChannel(BC_NAME)
        bc.postMessage({ type: 'changed', at: Date.now() })
        bc.close()
    } catch {}

    // Fallback for browsers/environments without BroadcastChannel.
    try {
        localStorage.setItem(STORAGE_KEY, String(Date.now()))
    } catch {}
}

export function subscribeIngressDirectoryChanged(onChanged: () => void) {
    let bc: BroadcastChannel | null = null

    try {
        bc = new BroadcastChannel(BC_NAME)
        bc.onmessage = event => {
            if (event?.data?.type !== 'changed') return
            onChanged()
        }
    } catch {
        bc = null
    }

    const onStorage = (e: StorageEvent) => {
        if (e.key !== STORAGE_KEY) return
        onChanged()
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

