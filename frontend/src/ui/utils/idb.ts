// Lightweight IndexedDB helpers for persisting:
// - ingress directory handle
// - job detail cache (settings store)
// - local sha<->path(s) mapping for files inside the ingress folder

const DB_NAME = 'windsurf-analysis'
// Bump when adding/changing stores.
// NOTE: If a version was previously bumped without creating a store (e.g. during dev/HMR),
// users can end up with a DB at that version missing the store. Bumping again fixes it.
const DB_VERSION = 5
const STORE_SETTINGS = 'settings'
const STORE_INPROGRESS = 'inprogress'
const STORE_SHA_TO_PATH = 'sha_to_path'
const STORE_PATH_TO_SHA = 'path_to_sha'
const STORE_THUMBNAILS = 'thumbnails'

type SettingsRecord = { key: string; value: any }
type ThumbnailRecord = { key: string; blob: Blob; createdAt: number }

// NOTE: Thumbnail caching is persisted in IndexedDB (no in-memory layer).

function idbRequest<T = any>(req: IDBRequest<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        req.onsuccess = () => resolve(req.result)
        req.onerror = () => reject(req.error)
    })
}

function idbTxDone(tx: IDBTransaction): Promise<void> {
    return new Promise((resolve, reject) => {
        tx.oncomplete = () => resolve()
        tx.onerror = () => reject(tx.error)
        tx.onabort = () => reject(tx.error)
    })
}

function openDb(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION)
        req.onupgradeneeded = () => {
            const db = req.result
            if (!db.objectStoreNames.contains(STORE_SETTINGS)) {
                db.createObjectStore(STORE_SETTINGS, { keyPath: 'key' })
            }
            if (!db.objectStoreNames.contains(STORE_INPROGRESS)) {
                db.createObjectStore(STORE_INPROGRESS, { keyPath: 'hash' })
            }
            if (!db.objectStoreNames.contains(STORE_SHA_TO_PATH)) {
                db.createObjectStore(STORE_SHA_TO_PATH, { keyPath: 'sha' })
            }
            if (!db.objectStoreNames.contains(STORE_PATH_TO_SHA)) {
                db.createObjectStore(STORE_PATH_TO_SHA, { keyPath: 'path' })
            }
            if (!db.objectStoreNames.contains(STORE_THUMBNAILS)) {
                db.createObjectStore(STORE_THUMBNAILS, { keyPath: 'key' })
            }
        }
        req.onsuccess = () => {
            const db = req.result
            // If we ever bump DB_VERSION in a later build, close old connections so upgrades can proceed.
            db.onversionchange = () => {
                try {
                    db.close()
                } catch {}
            }
            resolve(db)
        }
        req.onerror = () => reject(req.error)
    })
}

export async function saveDirectoryHandle(handle: FileSystemDirectoryHandle): Promise<void> {
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_SETTINGS, 'readwrite')
        const store = tx.objectStore(STORE_SETTINGS)
        const rec: SettingsRecord = { key: 'ingressDir', value: handle }
        store.put(rec)
        await idbTxDone(tx)
        db.close()
    } catch {
        // Fallback for browsers that cannot persist handles (e.g., Safari)
    }
}

export async function loadDirectoryHandle(): Promise<FileSystemDirectoryHandle | null> {
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_SETTINGS, 'readonly')
        const store = tx.objectStore(STORE_SETTINGS)
        const rec = (await idbRequest(store.get('ingressDir'))) as SettingsRecord | undefined
        const value = (rec?.value as FileSystemDirectoryHandle) ?? null
        db.close()
        return value
    } catch {
        return null
    }
}

// Generic settings helpers
export async function saveSetting(key: string, value: any): Promise<void> {
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_SETTINGS, 'readwrite')
        const store = tx.objectStore(STORE_SETTINGS)
        const rec: SettingsRecord = { key, value }
        store.put(rec)
        await idbTxDone(tx)
        db.close()
    } catch {}
}

export async function loadSetting<T = any>(key: string): Promise<T | null> {
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_SETTINGS, 'readonly')
        const store = tx.objectStore(STORE_SETTINGS)
        const rec = (await idbRequest(store.get(key))) as SettingsRecord | undefined
        const value = (rec?.value as T) ?? null
        db.close()
        return value
    } catch {
        return null
    }
}

export async function deleteSetting(key: string): Promise<void> {
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_SETTINGS, 'readwrite')
        const store = tx.objectStore(STORE_SETTINGS)
        store.delete(key)
        await idbTxDone(tx)
        db.close()
    } catch {}
}

// SHA <-> Path mapping helpers
type ShaToPathRecord = { sha: string; paths: string[]; updatedAt: number }
type PathToShaRecord = { path: string; sha: string; updatedAt: number }

export type ShaPathMappingUpdate = {
    sha: string
    path: string
    prevPath: string | null
    prevShaForPath: string | null
}

const shaPathMappingListeners = new Set<(u: ShaPathMappingUpdate) => void>()

export function subscribeShaPathMappingUpdates(cb: (u: ShaPathMappingUpdate) => void): () => void {
    shaPathMappingListeners.add(cb)
    return () => shaPathMappingListeners.delete(cb)
}

function _normalizeRelativePath(p: string): string {
    return String(p || '')
        .replace(/^[./\\]+/, '')
        .replace(/\\/g, '/')
}

function _uniqSorted(paths: string[]): string[] {
    const out = Array.from(new Set(paths.filter(Boolean).map(_normalizeRelativePath)))
    out.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()))
    return out
}

function _extractPaths(rec: ShaToPathRecord | undefined | null): string[] {
    if (!rec) return []
    return Array.isArray(rec.paths) ? rec.paths.map(p => String(p || '')).filter(Boolean) : []
}

export async function saveShaPathMapping(sha: string, path: string): Promise<void> {
    try {
        const db = await openDb()
        await new Promise<void>((resolve, reject) => {
            const tx = db.transaction([STORE_SHA_TO_PATH, STORE_PATH_TO_SHA], 'readwrite')
            const shaStore = tx.objectStore(STORE_SHA_TO_PATH)
            const pathStore = tx.objectStore(STORE_PATH_TO_SHA)
            let prevPathsForSha: string[] = []
            let prevPath: string | null = null
            let prevShaForPath: string | null = null
            let changed = false
            let pending = 2

            const maybeFinish = () => {
                pending -= 1
                if (pending !== 0) return
                const shaNorm = String(sha || '').toLowerCase()
                const pathNorm = _normalizeRelativePath(path)
                if (!shaNorm || !pathNorm) return

                // One SHA can map to multiple paths (duplicate files). Keep a stable set.
                const nextPathsForSha = _uniqSorted([...prevPathsForSha, pathNorm])

                // If this path used to point to another SHA (file replaced), remove it from that old SHA's paths.
                if (prevShaForPath && String(prevShaForPath).toLowerCase() !== shaNorm) {
                    const oldSha = String(prevShaForPath).toLowerCase()
                    const oldShaGet = shaStore.get(oldSha)
                    oldShaGet.onsuccess = () => {
                        const oldRec = oldShaGet.result as ShaToPathRecord | undefined
                        const oldPaths = _extractPaths(oldRec)
                        const remaining = _uniqSorted(oldPaths.filter(p => _normalizeRelativePath(p) !== pathNorm))
                        if (remaining.length === 0) shaStore.delete(oldSha)
                        else {
                            const now = Date.now()
                            const orec: ShaToPathRecord = { sha: oldSha, paths: remaining, updatedAt: now }
                            shaStore.put(orec)
                        }
                    }
                }

                const now = Date.now()
                const srec: ShaToPathRecord = { sha: shaNorm, paths: nextPathsForSha, updatedAt: now }
                const prec: PathToShaRecord = { path: pathNorm, sha: shaNorm, updatedAt: now }
                shaStore.put(srec)
                pathStore.put(prec)
            }

            const shaGet = shaStore.get(sha)
            shaGet.onsuccess = () => {
                const rec = shaGet.result as ShaToPathRecord | undefined
                prevPathsForSha = _extractPaths(rec)
                prevPath = prevPathsForSha.length ? prevPathsForSha[0] : null
                if (!prevPathsForSha.some(p => _normalizeRelativePath(p) === _normalizeRelativePath(path)))
                    changed = true
                maybeFinish()
            }
            shaGet.onerror = () => {
                changed = true
                maybeFinish()
            }

            const pathGet = pathStore.get(path)
            pathGet.onsuccess = () => {
                prevShaForPath = (pathGet.result as PathToShaRecord | undefined)?.sha ?? null
                if ((prevShaForPath || '').toLowerCase() !== (sha || '').toLowerCase()) changed = true
                maybeFinish()
            }
            pathGet.onerror = () => {
                changed = true
                maybeFinish()
            }

            tx.oncomplete = () => {
                // If nothing changed, don't notify.
                if (changed) {
                    const update: ShaPathMappingUpdate = {
                        sha: String(sha || '').toLowerCase(),
                        path: _normalizeRelativePath(path),
                        prevPath,
                        prevShaForPath,
                    }
                    // Notify in-process subscribers (avoids global DOM event jank)
                    for (const cb of shaPathMappingListeners) {
                        try {
                            cb(update)
                        } catch {}
                    }
                }
                resolve()
            }
            tx.onerror = () => reject(tx.error)
        })
        db.close()
    } catch {}
}

export async function getPathForSha(sha: string): Promise<string | null> {
    try {
        const db = await openDb()
        const value = await new Promise<string | null>((resolve, reject) => {
            const tx = db.transaction(STORE_SHA_TO_PATH, 'readonly')
            const store = tx.objectStore(STORE_SHA_TO_PATH)
            const req = store.get(sha)
            req.onsuccess = () => {
                const rec = req.result as ShaToPathRecord | undefined
                const paths = _uniqSorted(_extractPaths(rec))
                resolve(paths.length ? paths[0] : null)
            }
            req.onerror = () => reject(req.error)
        })
        db.close()
        return value
    } catch {
        return null
    }
}

export async function getPathsForSha(sha: string): Promise<string[]> {
    try {
        const db = await openDb()
        const value = await new Promise<string[]>((resolve, reject) => {
            const tx = db.transaction(STORE_SHA_TO_PATH, 'readonly')
            const store = tx.objectStore(STORE_SHA_TO_PATH)
            const req = store.get(sha)
            req.onsuccess = () => {
                const rec = req.result as ShaToPathRecord | undefined
                resolve(_uniqSorted(_extractPaths(rec)))
            }
            req.onerror = () => reject(req.error)
        })
        db.close()
        return value
    } catch {
        return []
    }
}

export async function getShaForPath(path: string): Promise<string | null> {
    try {
        const db = await openDb()
        const value = await new Promise<string | null>((resolve, reject) => {
            const tx = db.transaction(STORE_PATH_TO_SHA, 'readonly')
            const store = tx.objectStore(STORE_PATH_TO_SHA)
            const req = store.get(path)
            req.onsuccess = () => resolve(((req.result as PathToShaRecord | undefined)?.sha as string) ?? null)
            req.onerror = () => reject(req.error)
        })
        db.close()
        return value
    } catch {
        return null
    }
}

/**
 * Remove stale sha/path mappings for files that no longer exist locally.
 *
 * This fixes "ghost folders" when users move/rename files while the analyzer is closed.
 * Call it with the current set of relative paths present in the ingress folder.
 */
export async function pruneShaPathMappings(existingPaths: Iterable<string>): Promise<void> {
    try {
        const validLower = new Set<string>()
        for (const p of existingPaths) validLower.add(_normalizeRelativePath(p).toLowerCase())

        const db = await openDb()
        const changedShas = new Set<string>()

        const tx = db.transaction([STORE_SHA_TO_PATH, STORE_PATH_TO_SHA], 'readwrite')
        const shaStore = tx.objectStore(STORE_SHA_TO_PATH)
        const pathStore = tx.objectStore(STORE_PATH_TO_SHA)

        // Cursor iteration is still event-based in IDB; keep it local and small.
        await new Promise<void>((resolve, reject) => {
            const cursorReq = pathStore.openCursor()
            cursorReq.onerror = () => reject(cursorReq.error)
            cursorReq.onsuccess = () => {
                const cursor = cursorReq.result as IDBCursorWithValue | null
                if (!cursor) return resolve()

                const rec = cursor.value as PathToShaRecord
                const pathNorm = _normalizeRelativePath(String(rec?.path || ''))
                const shaNorm = String(rec?.sha || '').toLowerCase()

                const keep = pathNorm && validLower.has(pathNorm.toLowerCase())
                if (keep) return cursor.continue()

                // Delete stale path->sha entry
                cursor.delete()

                // Remove this path from the sha->paths list (and delete sha record if it becomes empty)
                if (!shaNorm) return cursor.continue()

                const shaGet = shaStore.get(shaNorm)
                shaGet.onerror = () => cursor.continue()
                shaGet.onsuccess = () => {
                    const shaRec = shaGet.result as ShaToPathRecord | undefined
                    const remaining = _uniqSorted(
                        _extractPaths(shaRec).filter(
                            p => _normalizeRelativePath(p).toLowerCase() !== pathNorm.toLowerCase()
                        )
                    )
                    if (remaining.length === 0) shaStore.delete(shaNorm)
                    else
                        shaStore.put({
                            sha: shaNorm,
                            paths: remaining,
                            updatedAt: Date.now(),
                        } satisfies ShaToPathRecord)
                    changedShas.add(shaNorm)
                    cursor.continue()
                }
            }
        })

        await idbTxDone(tx)

        db.close()

        // Notify in-process subscribers so the UI refreshes immediately (e.g. removes ghost folders).
        for (const sha of changedShas) {
            const update: ShaPathMappingUpdate = { sha, path: '', prevPath: null, prevShaForPath: null }
            for (const cb of shaPathMappingListeners) {
                try {
                    cb(update)
                } catch {}
            }
        }
    } catch {}
}

// Thumbnail caching helpers (sha-based keys)
export async function getThumbnailBlob(key: string): Promise<Blob | null> {
    try {
        if (!key) return null
        const db = await openDb()
        try {
            const tx = db.transaction(STORE_THUMBNAILS, 'readonly')
            const store = tx.objectStore(STORE_THUMBNAILS)
            const rec = (await idbRequest(store.get(key))) as ThumbnailRecord | undefined
            await idbTxDone(tx)
            return (rec?.blob as Blob) ?? null
        } finally {
            db.close()
        }
    } catch {
        return null
    }
}

export async function saveThumbnailBlob(key: string, blob: Blob): Promise<void> {
    try {
        if (!key) return
        const db = await openDb()
        try {
            const tx = db.transaction(STORE_THUMBNAILS, 'readwrite')
            const store = tx.objectStore(STORE_THUMBNAILS)
            store.put({ key, blob, createdAt: Date.now() } satisfies ThumbnailRecord)
            await idbTxDone(tx)
        } finally {
            db.close()
        }
    } catch {}
}

export async function deleteThumbnailBlob(key: string): Promise<void> {
    try {
        if (!key) return
        const db = await openDb()
        try {
            const tx = db.transaction(STORE_THUMBNAILS, 'readwrite')
            const store = tx.objectStore(STORE_THUMBNAILS)
            store.delete(key)
            await idbTxDone(tx)
        } finally {
            db.close()
        }
    } catch {}
}
