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
            if (suspended) return false

            // Gold standard: if a job with this checksum already exists for this user, skip.
            if (knownChecksumsSha256?.has(shaLower)) return false

            // Short TTL cache to avoid repeated /jobs calls for the same checksum while Firestore updates.
            const lastHandledAt = recentlyHandledRef.current.get(shaLower) ?? 0
            const HANDLED_TTL_MS = 15_000
            if (Date.now() - lastHandledAt < HANDLED_TTL_MS) return false

            if (failedRef.current.has(shaLower)) return false
            if (inProgressRef.current.has(shaLower)) return false

            return true
        },
        [knownChecksumsSha256, suspended, uploadCtx, uploadsEnabled]
    )

    const uploadOne = React.useCallback(
        async (shaLower: string, file: File, relPath: string) => {
            setUploading(v => v + 1)
            setUploads(prev => {
                if (prev.some(u => u.id === shaLower)) return prev
                return [...prev, { id: shaLower, relativePath: relPath, progress: 0, status: 'queued', error: null }]
            })

            try {
                inProgressRef.current.add(shaLower)

                setLastError(null)
                updateUpload(shaLower, { status: 'uploading' })
                const result = await uploadVideoFile(file, settings.uploadQuality, uploadCtx!, percent =>
                    updateUpload(shaLower, { progress: Math.round(percent * 100) })
                )
                recentlyHandledRef.current.set(shaLower, Date.now())

                if (result === 'skipped') {
                    removeUpload(shaLower)
                    return
                }
                updateUpload(shaLower, { progress: 100, status: 'done' })
            } catch (e: any) {
                console.error('Upload failed for', relPath, e)
                setLastError(e?.message || String(e))
                updateUpload(shaLower, { status: 'error', error: e?.message || String(e) })
                failedRef.current.add(shaLower)
                setSuspended(true)
            } finally {
                inProgressRef.current.delete(shaLower)
                setUploading(v => v - 1)
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
            const PRUNE_EVERY_MS = 30_000
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
            if (hasRemovals || now - lastPruneAtRef.current > PRUNE_EVERY_MS) {
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
        setLastError(null)
        setUploads(prev =>
            prev.map(u => (u.status === 'error' ? { ...u, status: 'queued', progress: 0, error: null } : u))
        )
        setSuspended(false) // This will trigger a new scan
    }, [scanOnce])

    return { active, lastRunAt, lastError, uploading, uploads, suspended, retryFailed }
}
