import React from 'react'
import { UploadContext, uploadVideoFile } from '../utils/uploader'
import { useSettings } from './useSettings'
import { useLocalFileIndex } from './useLocalFileIndex'
import { fingerprintKey, getFingerprintSha } from '../utils/localFileIndex'

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

                const work: Promise<void>[] = []
                for (const fp of newFingerprints) {
                    const sha = getFingerprintSha(snapshot, fp)
                    if (!sha) continue
                    if (!shouldUpload(sha)) continue
                    const file = filesByKey.get(fingerprintKey(fp))
                    if (!file) continue
                    work.push(uploadOne(sha, file, fp.path))
                }
                await Promise.all(work)

            }

            setLastRunAt(Date.now())
        } catch (e: any) {
            setLastError(e?.message || String(e))
            setLastRunAt(Date.now())
        }
        timerRef.current = window.setTimeout(scanContinuously, intervalMs)
    }, [dirHandle, intervalMs, loaded, refresh, shouldUpload, uploadOne])

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
