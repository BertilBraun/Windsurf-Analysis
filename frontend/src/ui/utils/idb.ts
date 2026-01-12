import type { FileSnapshot } from './localFileIndex'

// Lightweight IndexedDB helpers for persisting:
// - ingress directory handle
// - job detail cache (settings store)
// - local file snapshot index for the ingress folder
// - in-progress ingress upload sessions (for resume + early job mapping)

const DB_NAME = 'windsurf-analysis'
// Bump when adding/changing stores.
// NOTE: If a version was previously bumped without creating a store (e.g. during dev/HMR),
// users can end up with a DB at that version missing the store. Bumping again fixes it.
const DB_VERSION = 12
const STORE_SETTINGS = 'settings'

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

// Generic settings helpers
export async function saveSetting(key: string, value: any): Promise<void> {
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_SETTINGS, 'readwrite')
        const store = tx.objectStore(STORE_SETTINGS)
        store.put({ key, value } satisfies { key: string; value: any })
        await idbTxDone(tx)
        db.close()
    } catch {}
}

export async function saveSettings(settings: Record<string, any>): Promise<void> {
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_SETTINGS, 'readwrite')
        const store = tx.objectStore(STORE_SETTINGS)
        for (const [key, value] of Object.entries(settings)) {
            store.put({ key, value } satisfies { key: string; value: any })
        }
        await idbTxDone(tx)
        db.close()
    } catch {}
}

export async function loadSetting<T = any>(key: string): Promise<T | null> {
    try {
        const db = await openDb()
        const tx = db.transaction(STORE_SETTINGS, 'readonly')
        const store = tx.objectStore(STORE_SETTINGS)
        const rec = (await idbRequest(store.get(key))) as { key: string; value: T } | undefined
        const value = rec?.value ?? null
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

export async function saveDirectoryHandle(handle: FileSystemDirectoryHandle): Promise<void> {
    return saveSetting('INGRESS_DIRECTORY_HANDLE', handle)
}

export async function loadDirectoryHandle(): Promise<FileSystemDirectoryHandle | null> {
    return loadSetting<FileSystemDirectoryHandle>('INGRESS_DIRECTORY_HANDLE')
}

export async function saveFileSnapshot(snapshot: FileSnapshot): Promise<void> {
    return saveSetting('FILE_SNAPSHOT', snapshot)
}

export async function loadFileSnapshot(): Promise<FileSnapshot | null> {
    return loadSetting<FileSnapshot>('FILE_SNAPSHOT')
}

// Last known local path per sha (used for unmapped job labels)
export async function saveLastKnownPaths(entries: Array<{ sha: string; path: string }>): Promise<void> {
    const settings: Record<string, string> = {}
    for (const entry of entries) {
        if (!entry.sha || !entry.path) continue
        settings['LAST_KNOWN_PATH_' + entry.sha] = entry.path
    }
    await saveSettings(settings)
}

export async function loadLastKnownPath(sha: string): Promise<string | null> {
    return loadSetting<string>('LAST_KNOWN_PATH_' + sha)
}

// Thumbnail caching helpers (sha-based keys)
export async function getThumbnailBlob(key: string): Promise<Blob | null> {
    return loadSetting('THUMBNAIL_' + key)
}

export async function saveThumbnailBlob(key: string, blob: Blob): Promise<void> {
    return saveSetting('THUMBNAIL_' + key, blob)
}

export async function deleteThumbnailBlob(key: string): Promise<void> {
    return deleteSetting('THUMBNAIL_' + key)
}
