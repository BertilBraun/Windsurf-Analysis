import React from 'react'
import { UploadContext, uploadVideoFile } from '../utils/uploader'
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
    uploadCtx: UploadContext | null,
    knownChecksumsSha256?: ReadonlySet<string> | null,
    pendingChecksumsSha256?: ReadonlySet<string> | null,
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
    const recentlyHandledRef = React.useRef<Map<string, number>>(new Map())
    // Track files that are still being copied so we only upload once they settle.
    const pendingStableRef = React.useRef<Set<string>>(new Set())
    const stabilityRef = React.useRef<Map<string, { size: number; mtimeMs: number; stableCount: number }>>(new Map())
    const pendingHandledRef = React.useRef<Set<string>>(new Set())
    const pendingChecksumsRef = React.useRef<ReadonlySet<string> | null>(null)
    const [suspended, setSuspended] = React.useState(false)
    const { settings } = useSettings()

    const updateUpload = React.useCallback((id: string, partial: Partial<IngressUploadItem>) => {
        setUploads(prev => prev.map(u => (u.id === id ? { ...u, ...partial } : u)))
    }, [])

    const removeUpload = React.useCallback((id: string) => {
        setUploads(prev => prev.filter(u => u.id !== id))
    }, [])

    const timerRef = React.useRef<number | null>(null)
    const { refresh, loaded, scanStatus } = useLocalFileIndex(dirHandle)

    React.useEffect(() => {
        pendingChecksumsRef.current = pendingChecksumsSha256 ?? null
        if (!pendingChecksumsSha256) {
            pendingHandledRef.current.clear()
            return
        }
        for (const sha of Array.from(pendingHandledRef.current)) {
            if (!pendingChecksumsSha256.has(sha)) pendingHandledRef.current.delete(sha)
        }
    }, [pendingChecksumsSha256])

    const shouldUpload = React.useCallback(
        (sha: string) => {
            if (!uploadsEnabled) return false
            if (!uploadCtx) return false

            // Gold standard: if a job with this checksum already exists for this user, skip.
            if (knownChecksumsSha256?.has(sha)) return false

            // Short TTL cache to avoid repeated /jobs calls for the same checksum while Firestore updates.
            const lastHandledAt = recentlyHandledRef.current.get(sha) ?? 0
            const HANDLED_TTL_MS = 15_000
            if (Date.now() - lastHandledAt < HANDLED_TTL_MS) return false

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
        [knownChecksumsSha256, uploadCtx, uploadsEnabled]
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
                    ctx: uploadCtx!,
                    onProgress: percent => updateUpload(sha, { progress: Math.round(percent * 100) }),
                    onStarted: () => updateUpload(sha, { status: 'uploading' }),
                    precomputedSha256: sha,
                })
                if (pendingChecksumsRef.current?.has(sha)) {
                    pendingHandledRef.current.add(sha)
                }
                recentlyHandledRef.current.set(sha, Date.now())

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
        [removeUpload, settings.uploadQuality, updateUpload, uploadCtx]
    )

    const scanContinuously = React.useCallback(async () => {
        if (!dirHandle) return
        if (!loaded) {
            timerRef.current = window.setTimeout(scanContinuously, intervalMs)
            return
        }

        try {
            const result = await refresh()
            if (result) {
                const { snapshot, filesByKey, newFingerprints } = result
                const stabilityByPath = new Map<string, boolean>()
                const REQUIRED_STABLE_SCANS = 2

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
                const pathToFingerprint = new Map(snapshot.files.map(fp => [fp.path, fp]))
                const currentPaths = new Set(snapshot.files.map(fp => fp.path))

                const ensureStable = (fp: FileFingerprint) => {
                    if (!isStable(fp)) {
                        pendingStableRef.current.add(fp.path)
                        return false
                    }
                    pendingStableRef.current.delete(fp.path)
                    return true
                }

                const queueUpload = (fp: FileFingerprint, sha: string) => {
                    if (!ensureStable(fp)) return
                    if (!shouldUpload(sha)) return
                    const file = filesByKey.get(fingerprintKey(fp))
                    if (!file) return
                    work.push(uploadOne(sha, file, fp.path))
                }

                for (const fp of newFingerprints) {
                    const sha = getFingerprintSha(snapshot, fp)
                    if (!sha) continue
                    queueUpload(fp, sha)
                }

                if (pendingStableRef.current.size > 0) {
                    for (const path of Array.from(pendingStableRef.current)) {
                        const fp = pathToFingerprint.get(path)
                        if (!fp) {
                            pendingStableRef.current.delete(path)
                            continue
                        }
                        const sha = getFingerprintSha(snapshot, fp)
                        if (!sha) continue
                        queueUpload(fp, sha)
                    }
                }

                if (pendingChecksumsSha256 && pendingChecksumsSha256.size > 0) {
                    for (const fp of snapshot.files) {
                        const sha = getFingerprintSha(snapshot, fp)
                        if (!sha) continue
                        if (!pendingChecksumsSha256.has(sha)) continue
                        if (pendingHandledRef.current.has(sha)) continue
                        queueUpload(fp, sha)
                    }
                }

                if (stabilityRef.current.size > 0) {
                    for (const path of Array.from(stabilityRef.current.keys())) {
                        if (!currentPaths.has(path)) {
                            stabilityRef.current.delete(path)
                            pendingStableRef.current.delete(path)
                        }
                    }
                }
                await Promise.all(work)
            }

            setLastRunAt(Date.now())
        } catch (e: any) {
            setLastError(e?.message || String(e))
            setLastRunAt(Date.now())
        }
        timerRef.current = window.setTimeout(scanContinuously, intervalMs)
    }, [dirHandle, intervalMs, loaded, refresh, shouldUpload, uploadOne, pendingChecksumsSha256])

    React.useEffect(() => {
        if (timerRef.current) window.clearTimeout(timerRef.current)
        if (!dirHandle) {
            setActive(false)
            return
        }
        setActive(true)
        // immediate run, then interval
        scanContinuously()
        return () => {
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
