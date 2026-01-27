import type { FileSnapshot } from './localFileIndex'

/**
 * @module idb
 * IndexedDB utility module for persisting application state, including:
 * - Ingress directory handles
 * - Job detail cache (settings store)
 * - Local file snapshot index
 * - Thumbnail blobs
 */

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

/**
 * Saves a single key-value pair to the IndexedDB settings store.
 * @param key - The unique identifier for the setting.
 * @param value - The value to persist.
 */
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

/**
 * Saves multiple key-value pairs to the IndexedDB settings store in a single transaction.
 * @param settings - A record of key-value pairs to persist.
 */
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

/**
 * Retrieves a value by key from the IndexedDB settings store.
 * @template T - The expected type of the value.
 * @param key - The unique identifier for the setting.
 * @returns The value if found, or null if not found or an error occurs.
 */
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

/**
 * Deletes a key-value pair from the IndexedDB settings store.
 * @param key - The unique identifier for the setting to remove.
 */
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

/**
 * Persists the FileSystemDirectoryHandle for the ingress folder.
 * @param handle - The directory handle to save.
 */
export async function saveDirectoryHandle(handle: FileSystemDirectoryHandle): Promise<void> {
    return saveSetting('INGRESS_DIRECTORY_HANDLE', handle)
}

/**
 * Loads the persisted FileSystemDirectoryHandle for the ingress folder.
 * @returns The directory handle if found, otherwise null.
 */
export async function loadDirectoryHandle(): Promise<FileSystemDirectoryHandle | null> {
    return loadSetting<FileSystemDirectoryHandle>('INGRESS_DIRECTORY_HANDLE')
}

/**
 * Persists a FileSnapshot index of the ingress folder.
 * @param snapshot - The file snapshot to save.
 */
export async function saveFileSnapshot(snapshot: FileSnapshot): Promise<void> {
    return saveSetting('FILE_SNAPSHOT', snapshot)
}

/**
 * Loads the persisted FileSnapshot index.
 * @returns The file snapshot if found, otherwise null.
 */
export async function loadFileSnapshot(): Promise<FileSnapshot | null> {
    return loadSetting<FileSnapshot>('FILE_SNAPSHOT')
}

/**
 * Persists a batch of SHA-to-path mappings, used for identifying unmapped job labels.
 * @param entries - An array of objects containing file SHA and its corresponding path.
 */
export async function saveLastKnownPaths(entries: Array<{ sha: string; path: string }>): Promise<void> {
    const settings: Record<string, string> = {}
    for (const entry of entries) {
        if (!entry.sha || !entry.path) continue
        settings['LAST_KNOWN_PATH_' + entry.sha] = entry.path
    }
    await saveSettings(settings)
}

/**
 * Loads the last known local path for a specific file SHA.
 * @param sha - The file SHA to look up.
 * @returns The path if found, otherwise null.
 */
export async function loadLastKnownPath(sha: string): Promise<string | null> {
    return loadSetting<string>('LAST_KNOWN_PATH_' + sha)
}

/**
 * Retrieves a cached thumbnail Blob by its key (typically a file SHA).
 * @param key - The unique identifier for the thumbnail (e.g., file SHA).
 * @returns The thumbnail Blob if found, otherwise null.
 */
export async function getThumbnailBlob(key: string): Promise<Blob | null> {
    return loadSetting('THUMBNAIL_' + key)
}

/**
 * Persists a thumbnail Blob to the cache.
 * @param key - The unique identifier for the thumbnail (e.g., file SHA).
 * @param blob - The thumbnail Blob to save.
 */
export async function saveThumbnailBlob(key: string, blob: Blob): Promise<void> {
    return saveSetting('THUMBNAIL_' + key, blob)
}

/**
 * Deletes a cached thumbnail Blob.
 * @param key - The unique identifier for the thumbnail to remove.
 */
export async function deleteThumbnailBlob(key: string): Promise<void> {
    return deleteSetting('THUMBNAIL_' + key)
}
