import React from 'react'
import { addProcessedHash, hasProcessedHash } from '../utils/idb'
import { UploadContext, uploadVideoFile, computeSha256 } from '../utils/uploader'
import { listFilesRecursively } from '../utils/fsAccess'

export type IngressUploadStatus = 'queued' | 'uploading' | 'done' | 'error' | 'skipped'

export type IngressUploadItem = {
    id: string // sha256 of the original file
    relativePath: string
    progress: number // 0-100
    status: IngressUploadStatus
    error?: string | null
}

export function useIngressScanner(
    dirHandle: FileSystemDirectoryHandle | null,
    uploadCtx: UploadContext | null,
    onUploaded: () => void,
    intervalMs: number = 10000
) {
    const [active, setActive] = React.useState(false)
    const [lastRunAt, setLastRunAt] = React.useState<number | null>(null)
    const [lastError, setLastError] = React.useState<string | null>(null)
    const [queued, setQueued] = React.useState(0)
    const [uploading, setUploading] = React.useState(0)
    const [uploads, setUploads] = React.useState<IngressUploadItem[]>([])
    const inProgressRef = React.useRef<Set<string>>(new Set())
    const failedRef = React.useRef<Set<string>>(new Set())
    const [suspended, setSuspended] = React.useState(false)

    const updateUpload = React.useCallback((id: string, partial: Partial<IngressUploadItem>) => {
        setUploads(prev => prev.map(u => (u.id === id ? { ...u, ...partial } : u)))
    }, [])

    const timerRef = React.useRef<number | null>(null)

    const scanOnce = React.useCallback(async () => {
        if (!dirHandle || !uploadCtx || suspended) return
        let queued = 0
        let uploading = 0
        try {
            const entries = await listFilesRecursively(dirHandle as any, ['.mp4'])
            for (const entry of entries) {
                const file = await entry.getFile()
                if (file.type.toLowerCase() !== 'video/mp4') continue
                const relPath = entry.relativePath
                const identifier = entry.relativePath
                // TODO const { sha256 } = await computeSha256(file)
                const already = await hasProcessedHash(identifier)
                if (already) continue
                if (failedRef.current.has(identifier)) continue
                if (inProgressRef.current.has(identifier)) continue

                queued += 1
                uploading += 1
                setUploading(v => v + 1)
                setQueued(v => v + 1)

                // add to uploads list as queued
                setUploads(prev => {
                    if (prev.some(u => u.id === identifier)) return prev
                    return [
                        ...prev,
                        { id: identifier, relativePath: relPath, progress: 0, status: 'queued', error: null },
                    ]
                })

                try {
                    // mark as in-progress (memory) and uploading
                    inProgressRef.current.add(identifier)
                    updateUpload(identifier, { status: 'uploading' })
                    await uploadVideoFile(
                        file,
                        uploadCtx,
                        percent => updateUpload(identifier, { progress: percent }),
                        relPath
                    )
                    // success: persist processed and mark done 100%
                    await addProcessedHash(identifier)
                    updateUpload(identifier, { progress: 100, status: 'done' })
                    onUploaded()
                } catch (e: any) {
                    console.error('Upload failed for', relPath, e)
                    setLastError(e?.message || String(e))
                    updateUpload(identifier, { status: 'error', error: e?.message || String(e) })
                    failedRef.current.add(identifier)
                    // Suspend further scanning until user resumes
                    setSuspended(true)
                    return
                } finally {
                    inProgressRef.current.delete(identifier)
                    uploading -= 1
                    setUploading(v => v - 1)
                }
            }
            setLastRunAt(Date.now())
            setQueued(0)
        } catch (e: any) {
            setLastError(e?.message || String(e))
            setLastRunAt(Date.now())
        }
    }, [dirHandle, uploadCtx, suspended])

    React.useEffect(() => {
        if (timerRef.current) window.clearInterval(timerRef.current)
        if (!dirHandle || !uploadCtx || suspended) {
            setActive(false)
            return
        }
        setActive(true)
        // immediate run, then interval
        scanOnce()
        timerRef.current = window.setInterval(scanOnce, intervalMs)
        return () => {
            if (timerRef.current) window.clearInterval(timerRef.current)
            timerRef.current = null
        }
    }, [dirHandle, uploadCtx, intervalMs, scanOnce, suspended])

    const resume = React.useCallback(() => {
        setSuspended(false)
        // kick an immediate scan
        setTimeout(() => {
            scanOnce()
        }, 0)
    }, [scanOnce])

    const retryFailed = React.useCallback(() => {
        failedRef.current = new Set()
        setUploads(prev => prev.map(u => (u.status === 'error' ? { ...u, status: 'queued', progress: 0 } : u)))
        resume()
    }, [resume])

    return { active, lastRunAt, lastError, queued, uploading, uploads, suspended, resume, retryFailed }
}
