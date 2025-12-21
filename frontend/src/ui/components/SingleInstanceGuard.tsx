import React from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from './Button'

const LOCK_KEY = 'windsurf:single:lock'
const HEARTBEAT_MS = 1000
const STALE_MS = 3000
const RECHECK_MS = 1000

function getNow() {
    return Date.now()
}

function readLock(): { id: string; ts: number } | null {
    try {
        const raw = localStorage.getItem(LOCK_KEY)
        if (!raw) return null
        return JSON.parse(raw)
    } catch {
        return null
    }
}

function writeLock(id: string) {
    try {
        localStorage.setItem(LOCK_KEY, JSON.stringify({ id, ts: getNow() }))
    } catch {}
}

function removeLock() {
    try {
        localStorage.removeItem(LOCK_KEY)
    } catch {}
}

function isDuplicateCheck(rec: { id: string; ts: number } | null, id: string) {
    return rec && getNow() - rec.ts < STALE_MS && rec.id !== id
}

export const SingleInstanceGuard: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { t } = useTranslation()
    const idRef = React.useRef<string>(`tab-${Math.random().toString(36).slice(2)}-${Date.now()}`)
    const [isDuplicate, setIsDuplicate] = React.useState(false)
    const leaderRef = React.useRef(false)
    const hbRef = React.useRef<number | null>(null)

    const tryBecomeLeader = React.useCallback(() => {
        const existing = readLock()
        if (isDuplicateCheck(existing, idRef.current)) {
            setIsDuplicate(true)
            leaderRef.current = false
            return
        }
        // attempt to acquire
        writeLock(idRef.current)
        const confirm = readLock()
        if (confirm && confirm.id === idRef.current) {
            leaderRef.current = true
            setIsDuplicate(false)
        } else {
            leaderRef.current = false
            setIsDuplicate(true)
        }
    }, [])

    React.useEffect(() => {
        tryBecomeLeader()
        const onStorage = (e: StorageEvent) => {
            if (e.key !== LOCK_KEY) return
            const rec = readLock()
            if (leaderRef.current) {
                // ignore others while we lead; they shouldn't overwrite our lock within heartbeat window
                return
            } else {
                const active = isDuplicateCheck(rec, idRef.current)
                setIsDuplicate(!!active)
            }
        }
        window.addEventListener('storage', onStorage)
        return () => {
            window.removeEventListener('storage', onStorage)
        }
    }, [tryBecomeLeader])

    // Recheck periodically when not leader in case the current leader disappears without cleanup.
    React.useEffect(() => {
        if (leaderRef.current) return
        const intervalId = window.setInterval(() => {
            if (!leaderRef.current) tryBecomeLeader()
        }, RECHECK_MS)
        return () => {
            window.clearInterval(intervalId)
        }
    }, [isDuplicate, tryBecomeLeader])

    // Heartbeat while leader
    React.useEffect(() => {
        if (!leaderRef.current) return
        const tick = () => writeLock(idRef.current)
        tick()
        hbRef.current = window.setInterval(tick, HEARTBEAT_MS)
        const onRelease = () => {
            if (leaderRef.current) removeLock()
        }
        window.addEventListener('beforeunload', onRelease)
        window.addEventListener('unload', onRelease)
        return () => {
            if (hbRef.current) window.clearInterval(hbRef.current)
            hbRef.current = null
            window.removeEventListener('beforeunload', onRelease)
            window.removeEventListener('unload', onRelease)
            // If this guard unmounts (e.g. user navigates away from Analyzer), release the lock.
            if (leaderRef.current) removeLock()
        }
    }, [isDuplicate])

    if (isDuplicate) {
        return (
            <div
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    minHeight: '100vh',
                    fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
                    background: '#f9fafb',
                    color: '#111827',
                    padding: 24,
                }}
            >
                <div
                    style={{
                        maxWidth: 560,
                        width: '100%',
                        background: 'white',
                        border: '1px solid #e5e7eb',
                        borderRadius: 8,
                        padding: 16,
                        boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
                    }}
                >
                    <h2 style={{ margin: 0, marginBottom: 8 }}>{t('components.singleInstanceGuard.title')}</h2>
                    <p style={{ marginTop: 0, marginBottom: 12, color: '#374151', fontSize: 14 }}>
                        {t('components.singleInstanceGuard.body')}
                    </p>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <Button
                            variant="unstyled"
                            size="none"
                            onClick={tryBecomeLeader}
                            text={t('components.singleInstanceGuard.retry')}
                        />
                    </div>
                </div>
            </div>
        )
    }

    return <>{children}</>
}
