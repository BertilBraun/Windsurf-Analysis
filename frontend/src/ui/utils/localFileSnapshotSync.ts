import { createCrossTabSignal } from './crossTabChannel'

const signal = createCrossTabSignal({
    broadcastChannelName: 'windsurf:localFileSnapshot',
    storageKey: 'windsurf:localFileSnapshot:changedAt',
})

export function notifyLocalFileSnapshotChanged() {
    signal.notify()
}

export function subscribeLocalFileSnapshotChanged(onChanged: () => void) {
    return signal.subscribe(onChanged)
}
