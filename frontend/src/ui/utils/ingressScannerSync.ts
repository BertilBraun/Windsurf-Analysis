/**
 * @fileoverview Provides synchronization utilities for the ingress scanner across multiple browser tabs.
 * Uses cross-tab communication channels to share state and broadcast commands.
 */

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

/**
 * Shared state of the ingress scanner synchronized across tabs.
 */
export type IngressScannerSharedState = {
    /** The unique identifier of the tab currently leading the scanning process. */
    leaderTabId: string
    /** Indicates if the scanner is currently running. */
    active: boolean
    /** Timestamp of the last completed scan run. */
    lastRunAt: number | null
    /** Error message from the most recent failure, if any. */
    lastError: string | null
    /** The number of files currently in the process of being uploaded. */
    uploading: number
    /** List of individual file upload items and their statuses. */
    uploads: IngressUploadItem[]
    /** Indicates if the scanner has been manually or automatically suspended. */
    suspended: boolean
    /** Total count of files detected during the scan. */
    detectedFiles: number
    /** The current progress or status of the local file indexer. */
    scanStatus: LocalFileIndexScanStatus
}

/**
 * Broadcasts the current scanner state to all other tabs.
 * @param state The state object to publish.
 */
export function publishIngressScannerState(state: IngressScannerSharedState) {
    stateChannel.publish({ type: 'state', state })
}

/**
 * Sends a command to all tabs to retry any failed upload operations.
 */
export function requestIngressRetryFailed() {
    cmdChannel.publish({ type: 'cmd', cmd: 'retryFailed', at: Date.now() })
}

/**
 * Subscribes to state updates from the ingress scanner.
 * @param onState Callback function triggered when a new state is received.
 * @returns A cleanup function to unsubscribe.
 */
export function subscribeIngressScannerState(onState: (state: IngressScannerSharedState) => void) {
    return stateChannel.subscribe(msg => {
        if (msg.type !== 'state') return
        onState(msg.state)
    })
}

/**
 * Subscribes to control commands for the ingress scanner.
 * @param onRetryFailed Callback function triggered when a retry command is received.
 * @returns A cleanup function to unsubscribe.
 */
export function subscribeIngressScannerCommands(onRetryFailed: () => void) {
    return cmdChannel.subscribe(msg => {
        if (msg.type !== 'cmd') return
        if (msg.cmd === 'retryFailed') onRetryFailed()
    })
}
