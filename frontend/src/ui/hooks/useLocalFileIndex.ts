import React from 'react'
import { FsEntry, listFilesRecursively } from '../utils/fsAccess'
import { loadFileSnapshot, saveFileSnapshot, saveLastKnownPaths } from '../utils/idb'
import {
    type FileFingerprint,
    type FileSnapshot,
    buildShaToPaths,
    computeSha256,
    fingerprintKey,
    normalizeRelativePath,
} from '../utils/localFileIndex'

type RefreshResult = {
    snapshot: FileSnapshot
    filesByKey: Map<string, File>
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
    const filesByKey = new Map<string, File>()
    const total = entries.length
    let processed = 0
    let lastUpdate = now()

    const maybeUpdate = () => {
        if (processed === total || now() - lastUpdate > 150 || processed % 25 === 0) {
            onProgress?.({ phase: 'hashing', total, processed })
            lastUpdate = now()
        }
    }

    const computeSha = async (entry: FsEntry) => {
        const file = await entry.getFile()
        if (file.type && file.type.toLowerCase() !== 'video/mp4') return
        const fp: FileFingerprint = {
            path: normalizeRelativePath(entry.relativePath),
            size: file.size,
            mtimeMs: file.lastModified,
        }
        const key = fingerprintKey(fp)
        filesByKey.set(key, file)
        files.push(fp)

        let sha = prevMap.get(key)
        if (!sha) {
            const computed = await computeSha256(file)
            sha = computed.sha256
        }
        fingerprintToSha[key] = String(sha)
        processed += 1
        maybeUpdate()
        if (processed % 25 === 0) await new Promise(resolve => window.setTimeout(resolve, 0))
    }

    await Promise.all(entries.map(computeSha))

    const snapshot: FileSnapshot = {
        files,
        fingerprintToSha,
        updatedAt: Date.now(),
    }

    return { snapshot, filesByKey }
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

    const shaToPaths = React.useMemo(() => buildShaToPaths(snapshot), [snapshot])

    return { snapshot, refresh, shaToPaths, loaded, scanStatus }
}
