import type { FileSnapshot } from './localFileIndex'

// Lightweight IndexedDB helpers for persisting:
// - ingress directory handle
// - job detail cache (settings store)
// - local file snapshot index for the ingress folder

const DB_NAME = 'windsurf-analysis'
// Bump when adding/changing stores.
// NOTE: If a version was previously bumped without creating a store (e.g. during dev/HMR),
// users can end up with a DB at that version missing the store. Bumping again fixes it.
const DB_VERSION = 9
const STORE_SETTINGS = 'settings'
const STORE_FILE_INDEX = 'file_index'
const STORE_LAST_KNOWN_PATH = 'last_known_path'
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
            if (!db.objectStoreNames.contains(STORE_FILE_INDEX)) {
                db.createObjectStore(STORE_FILE_INDEX, { keyPath: 'key' })
            }
            if (!db.objectStoreNames.contains(STORE_LAST_KNOWN_PATH)) {
                db.createObjectStore(STORE_LAST_KNOWN_PATH, { keyPath: 'sha' })
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

// File snapshot helpers
type FileSnapshotRecord = { key: string; value: FileSnapshot }

const FILE_SNAPSHOT_KEY = 'ingress_snapshot'

export async function saveFileSnapshot(snapshot: FileSnapshot): Promise<void> {
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_FILE_INDEX, 'readwrite')
        const store = tx.objectStore(STORE_FILE_INDEX)
        const rec: FileSnapshotRecord = { key: FILE_SNAPSHOT_KEY, value: snapshot }
        store.put(rec)
        await idbTxDone(tx)
        db.close()
    } catch {}
}

export async function loadFileSnapshot(): Promise<FileSnapshot | null> {
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_FILE_INDEX, 'readonly')
        const store = tx.objectStore(STORE_FILE_INDEX)
        const rec = (await idbRequest(store.get(FILE_SNAPSHOT_KEY))) as FileSnapshotRecord | undefined
        const value = (rec?.value as FileSnapshot) ?? null
        db.close()
        return value
    } catch {
        return null
    }
}

// Last known local path per sha (used for unmapped job labels)
type LastKnownPathRecord = { sha: string; path: string; updatedAt: number }

export async function saveLastKnownPaths(entries: Array<{ sha: string; path: string }>): Promise<void> {
    if (!entries.length) return
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_LAST_KNOWN_PATH, 'readwrite')
        const store = tx.objectStore(STORE_LAST_KNOWN_PATH)
        const now = Date.now()
        for (const entry of entries) {
            if (!entry.sha || !entry.path) continue
            store.put({ sha: entry.sha, path: entry.path, updatedAt: now } satisfies LastKnownPathRecord)
        }
        await idbTxDone(tx)
        db.close()
    } catch {}
}

export async function loadLastKnownPath(sha: string): Promise<string | null> {
    try {
        if (!sha) return null
        const db = await openDb()
        const tx = db.transaction(STORE_LAST_KNOWN_PATH, 'readonly')
        const store = tx.objectStore(STORE_LAST_KNOWN_PATH)
        const rec = (await idbRequest(store.get(sha))) as LastKnownPathRecord | undefined
        const value = (rec?.path as string) ?? null
        await idbTxDone(tx)
        db.close()
        return value
    } catch {
        return null
    }
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
