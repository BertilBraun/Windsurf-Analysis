const BC_NAME = 'windsurf:localFileSnapshot'
const STORAGE_KEY = 'windsurf:localFileSnapshot:changedAt'

export function notifyLocalFileSnapshotChanged() {
    try {
        const bc = new BroadcastChannel(BC_NAME)
        bc.postMessage({ type: 'changed', at: Date.now() })
        bc.close()
    } catch {}

    try {
        localStorage.setItem(STORAGE_KEY, String(Date.now()))
    } catch {}
}

export function subscribeLocalFileSnapshotChanged(onChanged: () => void) {
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

