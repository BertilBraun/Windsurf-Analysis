import React from 'react'

type LockRecord = { id: string; ts: number }

function now() {
    return Date.now()
}

function getOrCreateTabId() {
    const c = globalThis.crypto as Crypto | undefined
    if (c?.randomUUID) return `tab-${c.randomUUID()}`
    return `tab-${Math.random().toString(36).slice(2)}-${Date.now()}`
}

function safeJsonParse<T>(raw: string): T | null {
    try {
        return JSON.parse(raw) as T
    } catch {
        return null
    }
}

function readLock(lockKey: string): LockRecord | null {
    try {
        const raw = localStorage.getItem(lockKey)
        if (!raw) return null
        return safeJsonParse<LockRecord>(raw)
    } catch {
        return null
    }
}

function writeLock(lockKey: string, id: string) {
    try {
        localStorage.setItem(lockKey, JSON.stringify({ id, ts: now() }))
    } catch {}
}

function removeLock(lockKey: string) {
    try {
        localStorage.removeItem(lockKey)
    } catch {}
}

function isStale(rec: LockRecord | null, staleMs: number) {
    return !rec || now() - rec.ts >= staleMs
}

export function useTabLeader(
    lockKey: string,
    { heartbeatMs = 1000, staleMs = 3000, recheckMs = 1000 }: { heartbeatMs?: number; staleMs?: number; recheckMs?: number } = {}
) {
    const idRef = React.useRef<string>(getOrCreateTabId())
    const [isLeader, setIsLeader] = React.useState(false)
    const isLeaderRef = React.useRef(false)

    const updateLeaderState = React.useCallback(
        (next: boolean) => {
            isLeaderRef.current = next
            setIsLeader(next)
        },
        [setIsLeader]
    )

    const tryBecomeLeader = React.useCallback(() => {
        const existing = readLock(lockKey)

        if (!isStale(existing, staleMs) && existing?.id !== idRef.current) {
            updateLeaderState(false)
            return
        }

        writeLock(lockKey, idRef.current)
        // Re-read to avoid temporary split-brain during races.
        const confirm = readLock(lockKey)
        updateLeaderState(!!confirm && confirm.id === idRef.current)
    }, [lockKey, staleMs, updateLeaderState])

    // Initial acquire + storage event sync
    React.useEffect(() => {
        tryBecomeLeader()

        const onStorage = (e: StorageEvent) => {
            if (e.key !== lockKey) return
            const rec = readLock(lockKey)
            if (!rec) {
                updateLeaderState(false)
                return
            }
            updateLeaderState(rec.id === idRef.current)
        }

        window.addEventListener('storage', onStorage)
        return () => {
            window.removeEventListener('storage', onStorage)
        }
    }, [lockKey, tryBecomeLeader, updateLeaderState])

    // Heartbeat while leader; yield immediately if the lock is lost.
    React.useEffect(() => {
        if (!isLeader) return

        const tick = () => {
            writeLock(lockKey, idRef.current)
            const rec = readLock(lockKey)
            if (rec && rec.id !== idRef.current && !isStale(rec, staleMs)) updateLeaderState(false)
        }

        tick()
        const intervalId = window.setInterval(tick, heartbeatMs)

        const onRelease = () => {
            if (!isLeaderRef.current) return
            removeLock(lockKey)
        }

        window.addEventListener('beforeunload', onRelease)
        window.addEventListener('unload', onRelease)

        return () => {
            window.clearInterval(intervalId)
            window.removeEventListener('beforeunload', onRelease)
            window.removeEventListener('unload', onRelease)
            // If unmounted (e.g. navigate away), release the lock.
            if (isLeaderRef.current) removeLock(lockKey)
        }
    }, [heartbeatMs, isLeader, lockKey, staleMs, updateLeaderState])

    // Recheck periodically when not leader in case the leader disappears.
    React.useEffect(() => {
        if (isLeader) return
        const intervalId = window.setInterval(() => {
            if (!isLeaderRef.current) tryBecomeLeader()
        }, recheckMs)
        return () => window.clearInterval(intervalId)
    }, [isLeader, recheckMs, tryBecomeLeader])

    return { isLeader, tabId: idRef.current, tryBecomeLeader }
}
