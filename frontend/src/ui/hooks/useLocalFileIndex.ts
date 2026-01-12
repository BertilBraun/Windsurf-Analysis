import React from 'react'
import { FsEntry, listFilesRecursively } from '../utils/fsAccess'
import { loadFileSnapshot, saveFileSnapshot, saveLastKnownPaths } from '../utils/idb'
import {
    type FileFingerprint,
    type FileSnapshot,
    buildShaToPaths,
    computeSha256,
    normalizeRelativePath,
} from '../utils/localFileIndex'
import { mapLimit } from '../utils/concurrency'
import { assert } from '../utils/assert'

const PENDING_SHA256 = 'PENDING'
const FILE_EXTENSIONS = ['.mp4']

export type LocalFileIndexScanStatus = {
    phase: 'idle' | 'listing' | 'hashing' | 'saving'
    total: number
    processed: number
}

const snapshotListeners = new Set<(snapshot: FileSnapshot | null) => void>()

function emitSnapshotUpdate(snapshot: FileSnapshot | null) {
    for (const listener of snapshotListeners) {
        try {
            listener(snapshot)
        } catch {}
    }
}

async function scanDirectory(
    dirHandle: FileSystemDirectoryHandle,
    prevSnapshot: FileSnapshot | null,
    onProgress: (status: LocalFileIndexScanStatus) => void
): Promise<{ snapshot: FileSnapshot; getFileForFingerprint: (fingerprint: FileFingerprint) => Promise<File> }> {
    onProgress({ phase: 'listing', total: 0, processed: 0 })
    const entries = await listFilesRecursively(dirHandle, FILE_EXTENSIONS)

    const fileFingerprints: FileFingerprint[] = []
    const pathToEntry = new Map<string, FsEntry>()

    const readFingerprint = async (entry: FsEntry) => {
        try {
            const file = await entry.getFile()
            if (file.type && file.type.toLowerCase() !== 'video/mp4') return

            const fingerprint: FileFingerprint = {
                path: normalizeRelativePath(entry.relativePath),
                size: file.size,
                mtimeMs: file.lastModified,
                sha256: PENDING_SHA256,
            }
            pathToEntry.set(fingerprint.path, entry)

            const previousFingerprint = prevSnapshot?.fileFingerprints.find(
                fp => fp.path === fingerprint.path && fp.size === fingerprint.size && fp.mtimeMs === fingerprint.mtimeMs
            )
            if (previousFingerprint) {
                fingerprint.sha256 = previousFingerprint.sha256
            }

            fileFingerprints.push(fingerprint)
        } catch {
            // Ignore files that are transiently inaccessible (e.g. mid-copy).
        } finally {
            onProgress({ phase: 'listing', total: entries.length, processed: fileFingerprints.length })
        }
    }

    await mapLimit(entries, readFingerprint)

    const getFileForFingerprint = async (fingerprint: FileFingerprint): Promise<File> => {
        const entry = pathToEntry.get(fingerprint.path)
        assert(entry !== undefined, 'Entry should be found')
        const file = await entry!.getFile()
        assert(file.type.toLowerCase() === 'video/mp4', 'File should be a video')
        return file
    }

    // Compute the remaining sha256s
    const unComputedFingerprints = fileFingerprints.filter(fp => fp.sha256 === PENDING_SHA256)
    let processed = 0

    const computeFingerprintSha = async (fingerprint: FileFingerprint) => {
        fingerprint.sha256 = await computeSha256(await getFileForFingerprint(fingerprint))
        processed++
        onProgress({ phase: 'hashing', total: unComputedFingerprints.length, processed })
    }

    await mapLimit(unComputedFingerprints, computeFingerprintSha)

    assert(!fileFingerprints.some(fp => fp.sha256 === PENDING_SHA256), 'All fingerprints should have a sha256')

    return {
        snapshot: {
            fileFingerprints,
            updatedAt: Date.now(),
        },
        getFileForFingerprint,
    }
}

async function updateSavedSnapshot(snapshot: FileSnapshot) {
    const shaToPaths = buildShaToPaths(snapshot)
    const lastKnownEntries: Array<{ sha: string; path: string }> = []
    for (const [sha, paths] of shaToPaths.entries()) {
        if (paths.length > 0) lastKnownEntries.push({ sha, path: paths[0] })
    }
    await saveLastKnownPaths(lastKnownEntries)
    await saveFileSnapshot(snapshot)
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
            emitSnapshotUpdate(loaded)
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

    const refresh = React.useCallback(async (): Promise<{
        snapshot: FileSnapshot
        getFileForFingerprint: (fingerprint: FileFingerprint) => Promise<File>
    } | null> => {
        if (!dirHandle || !loaded) return null
        const { snapshot, getFileForFingerprint } = await scanDirectory(dirHandle, snapshotRef.current, setScanStatus)
        setScanStatus({ phase: 'saving', total: 0, processed: 0 })
        await updateSavedSnapshot(snapshot)
        snapshotRef.current = snapshot
        setSnapshot(snapshot)
        emitSnapshotUpdate(snapshot)
        setScanStatus({ phase: 'idle', total: 0, processed: 0 })
        return { snapshot, getFileForFingerprint }
    }, [dirHandle, loaded])

    const shaToPaths = React.useMemo(() => buildShaToPaths(snapshot), [snapshot])

    return { snapshot, refresh, shaToPaths, loaded, scanStatus }
}
