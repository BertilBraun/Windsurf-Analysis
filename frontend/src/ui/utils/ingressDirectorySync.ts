/**
 * @fileoverview Provides synchronization for ingress directory changes across browser tabs.
 */

import { createCrossTabSignal } from './crossTabChannel'

const signal = createCrossTabSignal({
    broadcastChannelName: 'windsurf:ingressDirectory',
    storageKey: 'windsurf:ingressDirectory:changedAt',
})

/**
 * Triggers a notification across all tabs that the ingress directory has been updated.
 */
export function notifyIngressDirectoryChanged() {
    signal.notify()
}

/**
 * Subscribes to notifications for ingress directory changes.
 * @param onChanged - Callback function invoked when a change occurs.
 * @returns An unsubscribe function to stop listening for changes.
 */
export function subscribeIngressDirectoryChanged(onChanged: () => void) {
    return signal.subscribe(onChanged)
}
