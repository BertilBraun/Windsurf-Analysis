import React from 'react'
import { saveShaPathMapping, getShaForPath, pruneShaPathMappings } from '../utils/idb'
import { UploadContext, uploadVideoFile, computeSha256 } from '../utils/uploader'
import { listFilesRecursively } from '../utils/fsAccess'
import { useSettings } from './useSettings'

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
    intervalMs: number = 2000
) {
    const [active, setActive] = React.useState(false)
    const [lastRunAt, setLastRunAt] = React.useState<number | null>(null)
    const [lastError, setLastError] = React.useState<string | null>(null)
    const [uploading, setUploading] = React.useState(0)
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
    const lastPruneAtRef = React.useRef<number>(0)
    const lastPathsRef = React.useRef<Set<string> | null>(null)

    const syncMappingForFile = React.useCallback(async (file: File, relPath: string): Promise<string | null> => {
        // Compute or reuse sha for this path
        let sha = await getShaForPath(relPath)
        if (!sha) {
            const { sha256 } = await computeSha256(file)
            sha = sha256
        }
        if (!sha) return null

        // Always persist mapping (adds duplicates + normalizes + updates timestamps)
        await saveShaPathMapping(sha, relPath)
        return String(sha).toLowerCase()
    }, [])

    const shouldUpload = React.useCallback(
        (shaLower: string) => {
            if (!uploadsEnabled) return false
            if (!uploadCtx) return false

            // Gold standard: if a job with this checksum already exists for this user, skip.
            if (knownChecksumsSha256?.has(shaLower)) return false

            // Short TTL cache to avoid repeated /jobs calls for the same checksum while Firestore updates.
            const lastHandledAt = recentlyHandledRef.current.get(shaLower) ?? 0
            const HANDLED_TTL_MS = 15_000
            if (Date.now() - lastHandledAt < HANDLED_TTL_MS) return false

            if (failedRef.current.has(shaLower)) {
                const retryAfter = retryAfterRef.current.get(shaLower)
                if (!retryAfter || Date.now() < retryAfter) return false
                failedRef.current.delete(shaLower)
                retryAfterRef.current.delete(shaLower)
                if (failedRef.current.size === 0) setSuspended(false)
            }
            if (inProgressRef.current.has(shaLower)) return false

            return true
        },
        [knownChecksumsSha256, uploadCtx, uploadsEnabled]
    )

    const uploadOne = React.useCallback(
        async (shaLower: string, file: File, relPath: string) => {
            let started = false
            const markStarted = () => {
                if (started) return
                started = true
                setUploading(v => v + 1)
                updateUpload(shaLower, { status: 'uploading' })
            }
            setUploads(prev => {
                if (prev.some(u => u.id === shaLower)) return prev
                return [...prev, { id: shaLower, relativePath: relPath, progress: 0, status: 'queued', error: null }]
            })

            try {
                inProgressRef.current.add(shaLower)

                setLastError(null)
                const result = await uploadVideoFile(
                    file,
                    settings.uploadQuality,
                    uploadCtx!,
                    percent => updateUpload(shaLower, { progress: Math.round(percent * 100) }),
                    markStarted
                )
                recentlyHandledRef.current.set(shaLower, Date.now())

                if (result === 'skipped') {
                    removeUpload(shaLower)
                    return
                }
                updateUpload(shaLower, { progress: 100, status: 'done' })
            } catch (e: any) {
                console.error('Upload failed for', relPath, e)
                const message = e?.message || String(e)
                setLastError(message)
                updateUpload(shaLower, { status: 'error', error: message })
                failedRef.current.add(shaLower)
                if (isTransientUploadError(e)) {
                    retryAfterRef.current.set(shaLower, Date.now() + AUTO_RETRY_MS)
                } else {
                    retryAfterRef.current.delete(shaLower)
                }
                setSuspended(true)
            } finally {
                inProgressRef.current.delete(shaLower)
                if (started) setUploading(v => v - 1)
            }
        },
        [removeUpload, settings.uploadQuality, updateUpload, uploadCtx]
    )

    const scanOnce = React.useCallback(async () => {
        if (!dirHandle) return

        try {
            const entries = await listFilesRecursively(dirHandle, ['.mp4'])
            const existingPaths = new Set<string>()
            const work: Promise<void>[] = []
            for (const entry of entries) {
                const file = await entry.getFile()
                if (file.type.toLowerCase() !== 'video/mp4') continue
                existingPaths.add(entry.relativePath)

                work.push(
                    (async () => {
                        const shaLower = await syncMappingForFile(file, entry.relativePath)
                        if (!shaLower) return
                        if (!shouldUpload(shaLower)) return
                        await uploadOne(shaLower, file, entry.relativePath)
                    })()
                )
            }
            await Promise.all(work)

            // Prune stale mappings occasionally (prevents "ghost folders" after offline renames/moves).
            const now = Date.now()
            const PRUNE_EVERY_MS = Math.max(2000, intervalMs)
            let hasRemovals = false
            const lastPaths = lastPathsRef.current
            if (lastPaths) {
                for (const prev of lastPaths) {
                    if (!existingPaths.has(prev)) {
                        hasRemovals = true
                        break
                    }
                }
            }
            lastPathsRef.current = existingPaths
            if (!lastPaths || hasRemovals || now - lastPruneAtRef.current >= PRUNE_EVERY_MS) {
                lastPruneAtRef.current = now
                await pruneShaPathMappings(existingPaths)
            }

            setLastRunAt(Date.now())
        } catch (e: any) {
            setLastError(e?.message || String(e))
            setLastRunAt(Date.now())
        }
    }, [dirHandle, shouldUpload, syncMappingForFile, uploadOne])

    React.useEffect(() => {
        if (timerRef.current) window.clearInterval(timerRef.current)
        if (!dirHandle) {
            setActive(false)
            return
        }
        setActive(true)
        // immediate run, then interval
        scanOnce()
        timerRef.current = window.setInterval(scanOnce, intervalMs)
        return () => {
            if (timerRef.current) window.clearInterval(timerRef.current)
            timerRef.current = null
        }
    }, [dirHandle, intervalMs, scanOnce])

    const retryFailed = React.useCallback(() => {
        failedRef.current = new Set()
        retryAfterRef.current = new Map()
        setLastError(null)
        setUploads(prev =>
            prev.map(u => (u.status === 'error' ? { ...u, status: 'queued', progress: 0, error: null } : u))
        )
        setSuspended(false) // This will trigger a new scan
    }, [scanOnce])

    return { active, lastRunAt, lastError, uploading, uploads, suspended, retryFailed }
}
