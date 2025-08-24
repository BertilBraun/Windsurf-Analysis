// Lightweight IndexedDB helpers for persisting the ingress directory handle
// and a set of processed file hashes.

const DB_NAME = 'windsurf-analysis'
const DB_VERSION = 2
const STORE_SETTINGS = 'settings'
const STORE_PROCESSED = 'processed'
const STORE_INPROGRESS = 'inprogress'

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
