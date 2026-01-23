import type { IngressUploadItem } from '../hooks/useIngressScanner'
import type { LocalFileIndexScanStatus } from '../hooks/useLocalFileIndex'

const BC_NAME = 'windsurf:ingressScanner'
const STORAGE_STATE_KEY = 'windsurf:ingressScanner:state'
const STORAGE_CMD_KEY = 'windsurf:ingressScanner:cmd'

export type IngressScannerSharedState = {
    leaderTabId: string
    active: boolean
    lastRunAt: number | null
    lastError: string | null
    uploading: number
    uploads: IngressUploadItem[]
    suspended: boolean
    detectedFiles: number
    scanStatus: LocalFileIndexScanStatus
}

type Msg =
    | { type: 'state'; state: IngressScannerSharedState }
    | { type: 'cmd'; cmd: 'retryFailed'; at: number }

function safeJsonParse<T>(raw: string): T | null {
    try {
        return JSON.parse(raw) as T
    } catch {
        return null
    }
}

function postMessage(message: Msg) {
    try {
        const bc = new BroadcastChannel(BC_NAME)
        bc.postMessage(message)
        bc.close()
    } catch {}
}

export function publishIngressScannerState(state: IngressScannerSharedState) {
    postMessage({ type: 'state', state })
    try {
        localStorage.setItem(STORAGE_STATE_KEY, JSON.stringify({ at: Date.now(), state }))
    } catch {}
}

export function requestIngressRetryFailed() {
    postMessage({ type: 'cmd', cmd: 'retryFailed', at: Date.now() })
    try {
        localStorage.setItem(STORAGE_CMD_KEY, JSON.stringify({ at: Date.now(), cmd: 'retryFailed' }))
    } catch {}
}

export function subscribeIngressScannerState(onState: (state: IngressScannerSharedState) => void) {
    let bc: BroadcastChannel | null = null

    try {
        bc = new BroadcastChannel(BC_NAME)
        bc.onmessage = event => {
            const msg = event?.data as Msg | undefined
            if (msg?.type !== 'state') return
            onState(msg.state)
        }
    } catch {
        bc = null
    }

    const onStorage = (e: StorageEvent) => {
        if (e.key !== STORAGE_STATE_KEY) return
        const parsed = safeJsonParse<{ at: number; state: IngressScannerSharedState }>(String(e.newValue || ''))
        if (!parsed?.state) return
        onState(parsed.state)
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

export function subscribeIngressScannerCommands(onRetryFailed: () => void) {
    let bc: BroadcastChannel | null = null

    try {
        bc = new BroadcastChannel(BC_NAME)
        bc.onmessage = event => {
            const msg = event?.data as Msg | undefined
            if (msg?.type !== 'cmd') return
            if (msg.cmd === 'retryFailed') onRetryFailed()
        }
    } catch {
        bc = null
    }

    const onStorage = (e: StorageEvent) => {
        if (e.key !== STORAGE_CMD_KEY) return
        const parsed = safeJsonParse<{ at: number; cmd: 'retryFailed' }>(String(e.newValue || ''))
        if (parsed?.cmd !== 'retryFailed') return
        onRetryFailed()
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

