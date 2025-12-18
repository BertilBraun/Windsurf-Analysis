// Lightweight IndexedDB helpers for persisting the ingress directory handle
// and a set of processed file hashes.

const DB_NAME = 'windsurf-analysis'
const DB_VERSION = 3
const STORE_SETTINGS = 'settings'
const STORE_PROCESSED = 'processed'
const STORE_INPROGRESS = 'inprogress'
const STORE_SHA_TO_PATH = 'sha_to_path'
const STORE_PATH_TO_SHA = 'path_to_sha'

type SettingsRecord = { key: string; value: any }
type ProcessedRecord = { hash: string; createdAt: number }

function openDb(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION)
        req.onupgradeneeded = () => {
            const db = req.result
            if (!db.objectStoreNames.contains(STORE_SETTINGS)) {
                db.createObjectStore(STORE_SETTINGS, { keyPath: 'key' })
            }
            if (!db.objectStoreNames.contains(STORE_PROCESSED)) {
                db.createObjectStore(STORE_PROCESSED, { keyPath: 'hash' })
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
        }
        req.onsuccess = () => resolve(req.result)
        req.onerror = () => reject(req.error)
    })
}

export async function saveDirectoryHandle(handle: FileSystemDirectoryHandle): Promise<void> {
    try {
        const db = await openDb()
        await new Promise<void>((resolve, reject) => {
            const tx = db.transaction(STORE_SETTINGS, 'readwrite')
            const store = tx.objectStore(STORE_SETTINGS)
            const rec: SettingsRecord = { key: 'ingressDir', value: handle }
            store.put(rec)
            tx.oncomplete = () => resolve()
            tx.onerror = () => reject(tx.error)
        })
        db.close()
    } catch {
        // Fallback for browsers that cannot persist handles (e.g., Safari)
    }
}

export async function loadDirectoryHandle(): Promise<FileSystemDirectoryHandle | null> {
    try {
        const db = await openDb()
        const value = await new Promise<FileSystemDirectoryHandle | null>((resolve, reject) => {
            const tx = db.transaction(STORE_SETTINGS, 'readonly')
            const store = tx.objectStore(STORE_SETTINGS)
            const req = store.get('ingressDir')
            req.onsuccess = () => resolve((req.result as SettingsRecord | undefined)?.value ?? null)
            req.onerror = () => reject(req.error)
        })
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
        await new Promise<void>((resolve, reject) => {
            const tx = db.transaction(STORE_SETTINGS, 'readwrite')
            const store = tx.objectStore(STORE_SETTINGS)
            const rec: SettingsRecord = { key, value }
            store.put(rec)
            tx.oncomplete = () => resolve()
            tx.onerror = () => reject(tx.error)
        })
        db.close()
    } catch {}
}

export async function loadSetting<T = any>(key: string): Promise<T | null> {
    try {
        const db = await openDb()
        const value = await new Promise<T | null>((resolve, reject) => {
            const tx = db.transaction(STORE_SETTINGS, 'readonly')
            const store = tx.objectStore(STORE_SETTINGS)
            const req = store.get(key)
            req.onsuccess = () => resolve(((req.result as SettingsRecord | undefined)?.value as T) ?? null)
            req.onerror = () => reject(req.error)
        })
        db.close()
        return value
    } catch {
        return null
    }
}

export async function deleteSetting(key: string): Promise<void> {
    try {
        const db = await openDb()
        await new Promise<void>((resolve, reject) => {
            const tx = db.transaction(STORE_SETTINGS, 'readwrite')
            const store = tx.objectStore(STORE_SETTINGS)
            store.delete(key)
            tx.oncomplete = () => resolve()
            tx.onerror = () => reject(tx.error)
        })
        db.close()
    } catch {}
}

export async function addProcessedHash(hash: string): Promise<void> {
    try {
        const db = await openDb()
        await new Promise<void>((resolve, reject) => {
            const tx = db.transaction(STORE_PROCESSED, 'readwrite')
            const store = tx.objectStore(STORE_PROCESSED)
            const rec: ProcessedRecord = { hash, createdAt: Date.now() }
            store.put(rec)
            tx.oncomplete = () => resolve()
            tx.onerror = () => reject(tx.error)
        })
        db.close()
    } catch {}
}

export async function hasProcessedHash(hash: string): Promise<boolean> {
    try {
        const db = await openDb()
        const exists = await new Promise<boolean>((resolve, reject) => {
            const tx = db.transaction(STORE_PROCESSED, 'readonly')
            const store = tx.objectStore(STORE_PROCESSED)
            const req = store.get(hash)
            req.onsuccess = () => resolve(!!req.result)
            req.onerror = () => reject(req.error)
        })
        db.close()
        return exists
    } catch {
        return false
    }
}

export async function removeProcessedHash(hash: string): Promise<void> {
    try {
        const db = await openDb()
        await new Promise<void>((resolve, reject) => {
            const tx = db.transaction(STORE_PROCESSED, 'readwrite')
            const store = tx.objectStore(STORE_PROCESSED)
            store.delete(hash)
            tx.oncomplete = () => resolve()
            tx.onerror = () => reject(tx.error)
        })
        db.close()
    } catch {}
}

type InProgressRecord = { hash: string; owner: string; createdAt: number }

export async function tryClaimUpload(hash: string, owner: string, maxAgeMs: number = 60 * 60 * 1000): Promise<boolean> {
    try {
        const db = await openDb()
        const ok = await new Promise<boolean>((resolve, reject) => {
            const tx = db.transaction(STORE_INPROGRESS, 'readwrite')
            const store = tx.objectStore(STORE_INPROGRESS)
            const now = Date.now()
            const rec: InProgressRecord = { hash, owner, createdAt: now }
            const addReq = (store as any).add(rec)
            addReq.onsuccess = () => resolve(true)
            addReq.onerror = () => {
                // If exists, check staleness; if stale, replace
                const getReq = store.get(hash)
                getReq.onsuccess = () => {
                    const existing = getReq.result as InProgressRecord | undefined
                    if (existing && now - existing.createdAt > maxAgeMs) {
                        // stale: replace ownership
                        store.put(rec)
                        tx.oncomplete = () => resolve(true)
                        tx.onerror = () => reject(tx.error)
                    } else {
                        resolve(false)
                    }
                }
                getReq.onerror = () => resolve(false)
            }
        })
        db.close()
        return ok
    } catch {
        return false
    }
}

export async function releaseClaimUpload(hash: string, owner: string): Promise<void> {
    try {
        const db = await openDb()
        await new Promise<void>((resolve, reject) => {
            const tx = db.transaction(STORE_INPROGRESS, 'readwrite')
            const store = tx.objectStore(STORE_INPROGRESS)
            const getReq = store.get(hash)
            getReq.onsuccess = () => {
                const existing = getReq.result as InProgressRecord | undefined
                if (existing && existing.owner === owner) {
                    store.delete(hash)
                }
                tx.oncomplete = () => resolve()
                tx.onerror = () => reject(tx.error)
            }
            getReq.onerror = () => resolve()
        })
        db.close()
    } catch {}
}

// SHA <-> Path mapping helpers
type ShaToPathRecord = { sha: string; path: string; updatedAt: number }
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

export async function saveShaPathMapping(sha: string, path: string): Promise<void> {
    try {
        const db = await openDb()
        await new Promise<void>((resolve, reject) => {
            const tx = db.transaction([STORE_SHA_TO_PATH, STORE_PATH_TO_SHA], 'readwrite')
            const shaStore = tx.objectStore(STORE_SHA_TO_PATH)
            const pathStore = tx.objectStore(STORE_PATH_TO_SHA)
            let prevPath: string | null = null
            let prevShaForPath: string | null = null
            let changed = false
            let pending = 2

            const maybeFinish = () => {
                pending -= 1
                if (pending !== 0) return
                if (!changed) return

                // Keep the two stores consistent:
                // - If this SHA was previously mapped to a different path, remove the old path->sha entry.
                // - If this path was previously mapped to a different SHA, remove the old sha->path entry.
                //
                // This prevents stale mappings when files are moved back and forth.
                if (prevPath && prevPath !== path) {
                    const oldPath = prevPath
                    const oldPathGet = pathStore.get(oldPath)
                    oldPathGet.onsuccess = () => {
                        const existing = oldPathGet.result as PathToShaRecord | undefined
                        if (existing?.sha === sha) pathStore.delete(oldPath)
                    }
                }
                if (prevShaForPath && prevShaForPath !== sha) {
                    const oldSha = prevShaForPath
                    const oldShaGet = shaStore.get(oldSha)
                    oldShaGet.onsuccess = () => {
                        const existing = oldShaGet.result as ShaToPathRecord | undefined
                        if (existing?.path === path) shaStore.delete(oldSha)
                    }
                }

                const now = Date.now()
                const srec: ShaToPathRecord = { sha, path, updatedAt: now }
                const prec: PathToShaRecord = { path, sha, updatedAt: now }
                shaStore.put(srec)
                pathStore.put(prec)
            }

            const shaGet = shaStore.get(sha)
            shaGet.onsuccess = () => {
                prevPath = (shaGet.result as ShaToPathRecord | undefined)?.path ?? null
                if (prevPath !== path) changed = true
                maybeFinish()
            }
            shaGet.onerror = () => {
                changed = true
                maybeFinish()
            }

            const pathGet = pathStore.get(path)
            pathGet.onsuccess = () => {
                prevShaForPath = (pathGet.result as PathToShaRecord | undefined)?.sha ?? null
                if (prevShaForPath !== sha) changed = true
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
                        sha,
                        path,
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
            req.onsuccess = () => resolve(((req.result as ShaToPathRecord | undefined)?.path as string) ?? null)
            req.onerror = () => reject(req.error)
        })
        db.close()
        return value
    } catch {
        return null
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
