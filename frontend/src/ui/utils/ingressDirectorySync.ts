import { createCrossTabSignal } from './crossTabChannel'

const signal = createCrossTabSignal({
    broadcastChannelName: 'windsurf:ingressDirectory',
    storageKey: 'windsurf:ingressDirectory:changedAt',
})

export function notifyIngressDirectoryChanged() {
    signal.notify()
}

export function subscribeIngressDirectoryChanged(onChanged: () => void) {
    return signal.subscribe(onChanged)
}
