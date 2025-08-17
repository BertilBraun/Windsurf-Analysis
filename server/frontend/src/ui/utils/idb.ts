// Lightweight IndexedDB helpers for persisting the ingress directory handle
// and a set of processed file hashes.

const DB_NAME = 'windsurf-analysis'
const DB_VERSION = 1
const STORE_SETTINGS = 'settings'
const STORE_PROCESSED = 'processed'

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
