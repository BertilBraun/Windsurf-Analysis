import React from 'react'
import { AuthorizedFetch, uploadVideoFile } from '../utils/uploader'
import { useSettings } from './useSettings'
import { useLocalFileIndex } from './useLocalFileIndex'
import { fingerprintKey, getFingerprintSha, type FileFingerprint } from '../utils/localFileIndex'

export type IngressUploadStatus = 'queued' | 'uploading' | 'done' | 'error' | 'skipped'

export type IngressUploadItem = {
    id: string // sha256 of the original file
    relativePath: string
    progress: number // 0-100
    status: IngressUploadStatus
    error?: string | null
}

const AUTO_RETRY_MS = 15000
const HANDLED_TTL_MS = 15_000
const SKIPPED_TTL_MS = 60 * 60 * 1000

function isTransientUploadError(err: any): boolean {
    const msg = String(err?.message || err || '').toLowerCase()
    if (!msg) return false
    return (
        msg.includes('network') ||
        msg.includes('failed to fetch') ||
        msg.includes('timeout') ||
        msg.includes('timed out') ||
        msg.includes('http 5') ||
        msg.includes('502') ||
        msg.includes('503') ||
        msg.includes('504')
    )
}

export function useIngressScanner(
    dirHandle: FileSystemDirectoryHandle | null,
    authorizedFetch: AuthorizedFetch,
    knownChecksumsSha256: ReadonlySet<string> | null,
    pendingChecksumsSha256: ReadonlySet<string> | null,
    uploadsEnabled: boolean = true,
    intervalMs: number = 5000
) {
    const [active, setActive] = React.useState(false)
    const [lastRunAt, setLastRunAt] = React.useState<number | null>(null)
    const [lastError, setLastError] = React.useState<string | null>(null)
    const [uploads, setUploads] = React.useState<IngressUploadItem[]>([])
    const inProgressRef = React.useRef<Set<string>>(new Set())
    const failedRef = React.useRef<Set<string>>(new Set())
    const retryAfterRef = React.useRef<Map<string, number>>(new Map())
    // Avoid hammering the server for the same checksum between scans while waiting for
    // Firestore job snapshots to catch up. Keep TTL short so deletions/re-association work.
    const recentlyHandledUntilRef = React.useRef<Map<string, number>>(new Map())
    // Track files that are still being copied so we only upload once they settle.
    const stabilityRef = React.useRef<Map<string, { size: number; mtimeMs: number; stableCount: number }>>(new Map())
    const [suspended, setSuspended] = React.useState(false)
    const { settings } = useSettings()

    const updateUpload = React.useCallback((id: string, partial: Partial<IngressUploadItem>) => {
        setUploads(prev => prev.map(u => (u.id === id ? { ...u, ...partial } : u)))
    }, [])

    const removeUpload = React.useCallback((id: string) => {
        setUploads(prev => prev.filter(u => u.id !== id))
    }, [])

    const timerRef = React.useRef<number | null>(null)
    const scanGenRef = React.useRef<number>(0)
    const scanInFlightRef = React.useRef<boolean>(false)
    const { refresh, loaded, scanStatus } = useLocalFileIndex(dirHandle)

    const shouldUpload = React.useCallback(
        (sha: string) => {
            if (!uploadsEnabled) return false

            // Gold standard: if a job with this checksum already exists for this user, skip.
            if (knownChecksumsSha256?.has(sha)) return false
            // If the server already has an uploading job, don't create another upload attempt.
            if (pendingChecksumsSha256?.has(sha)) return false

            const handledUntil = recentlyHandledUntilRef.current.get(sha) ?? 0
            if (Date.now() < handledUntil) return false

            if (failedRef.current.has(sha)) {
                const retryAfter = retryAfterRef.current.get(sha)
                if (!retryAfter || Date.now() < retryAfter) return false
                failedRef.current.delete(sha)
                retryAfterRef.current.delete(sha)
                if (failedRef.current.size === 0) setSuspended(false)
            }
            if (inProgressRef.current.has(sha)) return false

            return true
        },
        [knownChecksumsSha256, pendingChecksumsSha256, uploadsEnabled]
    )

    const uploadOne = React.useCallback(
        async (sha: string, file: File, relPath: string) => {
            setUploads(prev => {
                if (prev.some(u => u.id === sha)) return prev
                return [...prev, { id: sha, relativePath: relPath, progress: 0, status: 'queued', error: null }]
            })

            try {
                inProgressRef.current.add(sha)

                const result = await uploadVideoFile({
                    file,
                    quality: settings.uploadQuality,
                    authorizedFetch,
                    onProgress: percent => updateUpload(sha, { progress: Math.round(percent * 100) }),
                    onStarted: () => updateUpload(sha, { status: 'uploading' }),
                    sha256: sha,
                })
                const ttl = result === 'skipped' ? SKIPPED_TTL_MS : HANDLED_TTL_MS
                recentlyHandledUntilRef.current.set(sha, Date.now() + ttl)

                if (result === 'skipped') {
                    removeUpload(sha)
                    return
                }
                updateUpload(sha, { progress: 100, status: 'done' })
                window.setTimeout(() => removeUpload(sha), 3000)
            } catch (e: any) {
                console.error('Upload failed for', relPath, e)
                const message = e?.message || String(e)
                setLastError(message)
                updateUpload(sha, { status: 'error', error: message })
                failedRef.current.add(sha)
                if (isTransientUploadError(e)) {
                    retryAfterRef.current.set(sha, Date.now() + AUTO_RETRY_MS)
                } else {
                    retryAfterRef.current.delete(sha)
                }
                setSuspended(true)
            } finally {
                inProgressRef.current.delete(sha)
            }
        },
        [removeUpload, settings.uploadQuality, updateUpload, authorizedFetch]
    )

    const scanContinuously = React.useCallback(async () => {
        const myGen = scanGenRef.current
        if (!dirHandle) return

        // Prevent overlapping scans; without this it's easy to end up with multiple concurrent
        // loops scheduling timers with different captured props/sets.
        if (scanInFlightRef.current || !loaded) {
            if (scanGenRef.current !== myGen) return
            timerRef.current = window.setTimeout(scanContinuously, intervalMs)
            return
        }

        scanInFlightRef.current = true
        try {
            const result = await refresh()
            if (result) {
                const { snapshot, filesByKey } = result

                const stabilityByPath = new Map<string, boolean>()
                const REQUIRED_STABLE_SCANS = 1

                const updateStability = (fp: FileFingerprint) => {
                    const prev = stabilityRef.current.get(fp.path)
                    if (prev && prev.size === fp.size && prev.mtimeMs === fp.mtimeMs) {
                        const stableCount = prev.stableCount + 1
                        stabilityRef.current.set(fp.path, { ...prev, stableCount })
                        return stableCount >= REQUIRED_STABLE_SCANS
                    }
                    stabilityRef.current.set(fp.path, { size: fp.size, mtimeMs: fp.mtimeMs, stableCount: 0 })
                    return false
                }

                const isStable = (fp: FileFingerprint) => {
                    const existing = stabilityByPath.get(fp.path)
                    if (typeof existing === 'boolean') return existing
                    const stable = updateStability(fp)
                    stabilityByPath.set(fp.path, stable)
                    return stable
                }

                const work: Promise<void>[] = []
                const currentPaths = new Set(snapshot.files.map(fp => fp.path))
                const queuedThisScan = new Set<string>()

                const queueUpload = (fp: FileFingerprint, sha: string) => {
                    if (!isStable(fp)) return
                    if (!shouldUpload(sha)) return
                    if (queuedThisScan.has(sha)) return
                    queuedThisScan.add(sha)
                    const file = filesByKey.get(fingerprintKey(fp))
                    if (!file) return
                    work.push(uploadOne(sha, file, fp.path))
                }

                for (const fp of snapshot.files) {
                    const sha = getFingerprintSha(snapshot, fp)
                    queueUpload(fp, sha)
                }

                if (stabilityRef.current.size > 0) {
                    for (const path of Array.from(stabilityRef.current.keys())) {
                        if (!currentPaths.has(path)) {
                            stabilityRef.current.delete(path)
                        }
                    }
                }
                await Promise.all(work)
            }

            setLastRunAt(Date.now())
        } catch (e: any) {
            setLastError(e?.message || String(e))
            setLastRunAt(Date.now())
        } finally {
            scanInFlightRef.current = false
        }
        if (scanGenRef.current !== myGen) return
        timerRef.current = window.setTimeout(scanContinuously, intervalMs)
    }, [dirHandle, intervalMs, loaded, refresh, shouldUpload, uploadOne])

    React.useEffect(() => {
        if (timerRef.current) window.clearTimeout(timerRef.current)
        if (!dirHandle) {
            setActive(false)
            return
        }
        setActive(true)
        scanGenRef.current += 1
        // immediate run, then interval
        scanContinuously()
        return () => {
            scanGenRef.current += 1
            if (timerRef.current) window.clearTimeout(timerRef.current)
            timerRef.current = null
        }
    }, [dirHandle, intervalMs, scanContinuously])

    const retryFailed = React.useCallback(() => {
        failedRef.current = new Set()
        retryAfterRef.current = new Map()
        setLastError(null)
        setUploads(prev =>
            prev.map(u => (u.status === 'error' ? { ...u, status: 'queued', progress: 0, error: null } : u))
        )
        setSuspended(false) // This will trigger a new scan
    }, [scanContinuously])

    const uploading = React.useMemo(
        () => uploads.filter(u => u.status === 'uploading' || u.status === 'queued').length,
        [uploads]
    )

    return { active, lastRunAt, lastError, uploading, uploads, suspended, retryFailed, scanStatus }
}
