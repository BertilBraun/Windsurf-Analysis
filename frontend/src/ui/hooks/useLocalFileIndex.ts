import React from 'react'
import { FsEntry, listFilesRecursively } from '../utils/fsAccess'
import { loadFileSnapshot, saveFileSnapshot, saveLastKnownPaths } from '../utils/idb'
import {
    type FileFingerprint,
    type FileSnapshot,
    buildShaToPaths,
    fingerprintKey,
    normalizeRelativePath,
} from '../utils/localFileIndex'

type RefreshResult = {
    snapshot: FileSnapshot
    entriesByKey: Map<string, FsEntry>
}

export type LocalFileIndexScanStatus = {
    phase: 'idle' | 'listing' | 'hashing' | 'saving'
    total: number
    processed: number
}

const snapshotListeners = new Set<(snapshot: FileSnapshot | null) => void>()

function emitSnapshotUpdate(snapshot: FileSnapshot | null) {
    for (const cb of snapshotListeners) {
        try {
            cb(snapshot)
        } catch {}
    }
}

function now() {
    return typeof performance !== 'undefined' ? performance.now() : Date.now()
}

function getMetadataConcurrency(): number {
    const hc =
        typeof navigator !== 'undefined' && (navigator as any).hardwareConcurrency
            ? Math.floor((navigator as any).hardwareConcurrency)
            : 0
    // Metadata reads are light; still keep it conservative to reduce filesystem contention.
    if (hc > 0) return Math.max(1, Math.min(6, Math.floor(hc / 2)))
    return 3
}

async function mapLimit<T>(items: T[], limit: number, fn: (item: T) => Promise<void>): Promise<void> {
    let nextIdx = 0
    async function worker(): Promise<void> {
        while (true) {
            const i = nextIdx
            if (i >= items.length) return
            nextIdx = i + 1
            await fn(items[i])
        }
    }
    const n = Math.max(1, Math.min(limit, items.length))
    await Promise.all(Array.from({ length: n }, () => worker()))
}

async function scanDirectory(
    dirHandle: FileSystemDirectoryHandle,
    prevSnapshot: FileSnapshot | null,
    extensions: string[],
    onProgress?: (status: LocalFileIndexScanStatus) => void
): Promise<RefreshResult> {
    onProgress?.({ phase: 'listing', total: 0, processed: 0 })
    const entries = await listFilesRecursively(dirHandle, extensions)
    const prevMap = new Map(Object.entries(prevSnapshot?.fingerprintToSha ?? {}))
    const files: FileFingerprint[] = []
    const fingerprintToSha: Record<string, string> = {}
    const entriesByKey = new Map<string, FsEntry>()
    const total = entries.length
    let processed = 0
    let lastUpdate = now()

    const maybeUpdate = () => {
        if (processed === total || now() - lastUpdate > 150 || processed % 25 === 0) {
            // Keep the existing "hashing" phase label for UI compatibility,
            // but we only read metadata here (no content hashing).
            onProgress?.({ phase: 'hashing', total, processed })
            lastUpdate = now()
        }
    }

    const readFingerprint = async (entry: FsEntry) => {
        try {
            const file = await entry.getFile()
            if (file.type && file.type.toLowerCase() !== 'video/mp4') return
            const fp: FileFingerprint = {
                path: normalizeRelativePath(entry.relativePath),
                size: file.size,
                mtimeMs: file.lastModified,
            }
            const key = fingerprintKey(fp)
            files.push(fp)
            entriesByKey.set(key, entry)

            const sha = prevMap.get(key)
            if (sha) fingerprintToSha[key] = String(sha)
        } catch {
            // Ignore files that are transiently inaccessible (e.g. mid-copy).
        } finally {
            processed += 1
            maybeUpdate()
            if (processed % 50 === 0) await new Promise(resolve => window.setTimeout(resolve, 0))
        }
    }

    await mapLimit(entries, getMetadataConcurrency(), readFingerprint)

    const snapshot: FileSnapshot = {
        files,
        fingerprintToSha,
        updatedAt: Date.now(),
    }

    return { snapshot, entriesByKey }
}

export function useLocalFileIndex(dirHandle: FileSystemDirectoryHandle | null) {
    const [snapshot, setSnapshot] = React.useState<FileSnapshot | null>(null)
    const [loaded, setLoaded] = React.useState(false)
    const [scanStatus, setScanStatus] = React.useState<LocalFileIndexScanStatus>({
        phase: 'idle',
        total: 0,
        processed: 0,
    })
    const snapshotRef = React.useRef<FileSnapshot | null>(null)

    React.useEffect(() => {
        let cancelled = false
        ;(async () => {
            const loaded = await loadFileSnapshot()
            if (cancelled) return
            snapshotRef.current = loaded
            setSnapshot(loaded)
            setLoaded(true)
        })()
        return () => {
            cancelled = true
        }
    }, [])

    React.useEffect(() => {
        // NOTE: this is used to update the state of two different useLocalFileIndex hooks after each refresh, so that not both have to perform the expensive refresh calculations
        const handler = (next: FileSnapshot | null) => {
            snapshotRef.current = next
            setSnapshot(next)
        }
        snapshotListeners.add(handler)
        return () => {
            snapshotListeners.delete(handler)
        }
    }, [])

    const refresh = React.useCallback(async (): Promise<RefreshResult | null> => {
        if (!dirHandle) return null
        const prevSnapshot = snapshotRef.current
        const result = await scanDirectory(dirHandle, prevSnapshot, ['.mp4'], setScanStatus)
        setScanStatus(prev => ({ ...prev, phase: 'saving' }))
        const shaToPaths = buildShaToPaths(result.snapshot)
        const lastKnownEntries: Array<{ sha: string; path: string }> = []
        for (const [sha, paths] of shaToPaths.entries()) {
            if (paths.length > 0) lastKnownEntries.push({ sha, path: paths[0] })
        }
        await saveLastKnownPaths(lastKnownEntries)
        await saveFileSnapshot(result.snapshot)
        snapshotRef.current = result.snapshot
        setSnapshot(result.snapshot)
        emitSnapshotUpdate(result.snapshot)
        setScanStatus({ phase: 'idle', total: 0, processed: 0 })
        return result
    }, [dirHandle])

    const rememberFingerprintSha = React.useCallback(async (fp: FileFingerprint, sha: string) => {
        const current = snapshotRef.current
        if (!current) return
        const key = fingerprintKey(fp)
        if (current.fingerprintToSha[key] === sha) return
        current.fingerprintToSha[key] = sha
        snapshotRef.current = current
        setSnapshot({ ...current, fingerprintToSha: { ...current.fingerprintToSha } })
        emitSnapshotUpdate(current)
        await saveFileSnapshot(current)
    }, [])

    const shaToPaths = React.useMemo(() => buildShaToPaths(snapshot), [snapshot])

    return { snapshot, refresh, rememberFingerprintSha, shaToPaths, loaded, scanStatus }
}
