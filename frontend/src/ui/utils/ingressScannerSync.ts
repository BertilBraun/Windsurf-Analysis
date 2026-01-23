import type { IngressUploadItem } from '../hooks/useIngressScanner'
import type { LocalFileIndexScanStatus } from '../hooks/useLocalFileIndex'
import { createCrossTabChannel } from './crossTabChannel'

const stateChannel = createCrossTabChannel<{ type: 'state'; state: IngressScannerSharedState }>({
    broadcastChannelName: 'windsurf:ingressScanner',
    storageKey: 'windsurf:ingressScanner:state',
})

const cmdChannel = createCrossTabChannel<{ type: 'cmd'; cmd: 'retryFailed'; at: number }>({
    broadcastChannelName: 'windsurf:ingressScanner',
    storageKey: 'windsurf:ingressScanner:cmd',
})

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

export function publishIngressScannerState(state: IngressScannerSharedState) {
    stateChannel.publish({ type: 'state', state })
}

export function requestIngressRetryFailed() {
    cmdChannel.publish({ type: 'cmd', cmd: 'retryFailed', at: Date.now() })
}

export function subscribeIngressScannerState(onState: (state: IngressScannerSharedState) => void) {
    return stateChannel.subscribe(msg => {
        if (msg.type !== 'state') return
        onState(msg.state)
    })
}

export function subscribeIngressScannerCommands(onRetryFailed: () => void) {
    return cmdChannel.subscribe(msg => {
        if (msg.type !== 'cmd') return
        if (msg.cmd === 'retryFailed') onRetryFailed()
    })
}
