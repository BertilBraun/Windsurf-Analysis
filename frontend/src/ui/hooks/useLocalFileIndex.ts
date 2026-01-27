/**
 * @file Provides a hook for indexing local files using the File System Access API.
 * Handles file discovery, hashing, and persistence of file snapshots.
 */

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
import { notifyLocalFileSnapshotChanged, subscribeLocalFileSnapshotChanged } from '../utils/localFileSnapshotSync'

const PENDING_SHA256 = 'PENDING'
const FILE_EXTENSIONS = ['.mp4']

/**
 * Represents the progress and state of a local file system scan.
 */
export type LocalFileIndexScanStatus = {
    /** The current stage of the indexing process. */
    phase: 'idle' | 'listing' | 'hashing' | 'saving'
    /** Total number of files or operations in the current phase. */
    total: number
    /** Number of files or operations completed in the current phase. */
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

/**
 * Manages a local file index for a directory handle.
 *
 * Scans for video files, computes SHA-256 fingerprints, and persists snapshots
 * to IndexedDB. Synchronizes state across hook instances and browser tabs.
 *
 * @param dirHandle - The directory handle to scan, or null if none is selected.
 * @returns An object containing:
 * - `snapshot`: The current file snapshot.
 * - `refresh`: Function to trigger a re-scan. Returns the new snapshot and a file retriever.
 * - `shaToPaths`: A map of SHA-256 hashes to relative file paths.
 * - `loaded`: Whether the initial snapshot has been loaded from storage.
 * - `scanStatus`: The current progress of the indexing operation.
 */
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
        loadFileSnapshot().then(loaded => {
            if (cancelled) return
            snapshotRef.current = loaded
            setSnapshot(loaded)
            emitSnapshotUpdate(loaded)
            setLoaded(true)
        })
        return () => {
            cancelled = true
        }
    }, [])

    // Keep snapshot in sync across tabs (used by useJobs for derived local paths).
    React.useEffect(() => {
        return subscribeLocalFileSnapshotChanged(() => {
            loadFileSnapshot().then(next => {
                if (next?.updatedAt && snapshotRef.current?.updatedAt === next.updatedAt) return
                snapshotRef.current = next
                setSnapshot(next)
                emitSnapshotUpdate(next)
            })
        })
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
        notifyLocalFileSnapshotChanged()
        setScanStatus({ phase: 'idle', total: 0, processed: 0 })
        return { snapshot, getFileForFingerprint }
    }, [dirHandle, loaded])

    const shaToPaths = React.useMemo(() => buildShaToPaths(snapshot), [snapshot])

    return { snapshot, refresh, shaToPaths, loaded, scanStatus }
}
