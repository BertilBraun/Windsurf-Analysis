import React from 'react'
import { addProcessedHash, hasProcessedHash } from '../utils/idb'
import { UploadContext, uploadVideoFile, computeSha256 } from '../utils/uploader'

function isProbablyVideoName(name: string): boolean {
    return /\.(mp4|mov|m4v|mkv|webm|avi|mts|m2ts)$/i.test(name)
}

export type ScannerState = {
    active: boolean
    lastRunAt: number | null
    lastError: string | null
    queued: number
    uploading: number
}

export function useIngressScanner(
    dirHandle: FileSystemDirectoryHandle | null,
    uploadCtx: UploadContext | null,
    intervalMs: number = 10000
) {
    const [state, setState] = React.useState<ScannerState>({
        active: false,
        lastRunAt: null,
        lastError: null,
        queued: 0,
        uploading: 0,
    })
    const timerRef = React.useRef<number | null>(null)

    const scanOnce = React.useCallback(async () => {
        if (!dirHandle || !uploadCtx) return
        let queued = 0
        let uploading = 0
        try {
            const walkAndUpload = async (directory: any, basePath: string) => {
                const hasEntries = typeof directory.entries === 'function'
                const hasValues = typeof directory.values === 'function'
                if (!hasEntries && !hasValues) return
                if (hasEntries) {
                    for await (const [name, handle] of directory.entries()) {
                        if (handle?.kind === 'directory') {
                            await walkAndUpload(handle, basePath ? `${basePath}${name}/` : `${name}/`)
                            continue
                        }
                        if (handle?.kind !== 'file') continue
                        if (!isProbablyVideoName(name)) continue
                        const file = await (handle as FileSystemFileHandle).getFile()
                        if (!file || !(file.type || '').startsWith('video/')) continue
                        const relPath = basePath ? `${basePath}${name}` : name
                        const { sha256 } = await computeSha256(file)
                        const already = await hasProcessedHash(sha256)
                        if (already) continue
                        queued += 1
                        setState(prev => ({ ...prev, queued }))
                        uploading += 1
                        setState(prev => ({ ...prev, uploading }))
                        try {
                            await uploadVideoFile(file, uploadCtx, undefined, relPath)
                            await addProcessedHash(sha256)
                        } catch (e: any) {
                            console.error('Upload failed for', relPath, e)
                            setState(prev => ({ ...prev, lastError: e?.message || String(e) }))
                        } finally {
                            uploading -= 1
                            setState(prev => ({ ...prev, uploading }))
                        }
                    }
                } else if (hasValues) {
                    for await (const handle of directory.values()) {
                        const name = (handle && (handle as any).name) || ''
                        if (handle?.kind === 'directory') {
                            await walkAndUpload(handle, basePath ? `${basePath}${name}/` : `${name}/`)
                            continue
                        }
                        if (handle?.kind !== 'file') continue
                        if (!isProbablyVideoName(name)) continue
                        const file = await (handle as FileSystemFileHandle).getFile()
                        if (!file || !(file.type || '').startsWith('video/')) continue
                        const relPath = basePath ? `${basePath}${name}` : name
                        const { sha256 } = await computeSha256(file)
                        const already = await hasProcessedHash(sha256)
                        if (already) continue
                        queued += 1
                        setState(prev => ({ ...prev, queued }))
                        uploading += 1
                        setState(prev => ({ ...prev, uploading }))
                        try {
                            await uploadVideoFile(file, uploadCtx, undefined, relPath)
                            await addProcessedHash(sha256)
                        } catch (e: any) {
                            console.error('Upload failed for', relPath, e)
                            setState(prev => ({ ...prev, lastError: e?.message || String(e) }))
                        } finally {
                            uploading -= 1
                            setState(prev => ({ ...prev, uploading }))
                        }
                    }
                }
            }

            await walkAndUpload(dirHandle as any, '')
            setState(prev => ({ ...prev, lastRunAt: Date.now(), queued: 0 }))
        } catch (e: any) {
            setState(prev => ({ ...prev, lastRunAt: Date.now(), lastError: e?.message || String(e) }))
        }
    }, [dirHandle, uploadCtx])

    React.useEffect(() => {
        if (timerRef.current) window.clearInterval(timerRef.current)
        if (!dirHandle || !uploadCtx) {
            setState(prev => ({ ...prev, active: false }))
            return
        }
        setState(prev => ({ ...prev, active: true }))
        // immediate run, then interval
        scanOnce()
        timerRef.current = window.setInterval(scanOnce, intervalMs)
        return () => {
            if (timerRef.current) window.clearInterval(timerRef.current)
            timerRef.current = null
        }
    }, [dirHandle, uploadCtx, intervalMs, scanOnce])

    return state
}
