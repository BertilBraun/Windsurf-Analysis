import React from 'react'
import { addProcessedHash, hasProcessedHash, removeProcessedHash } from '../utils/idb'
import { UploadContext, uploadVideoFile, computeSha256 } from '../utils/uploader'
import { listFilesRecursively, isMp4File, isMp4Name } from '../utils/fsAccess'

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

    const timerRef = React.useRef<number | null>(null)

    const scanOnce = React.useCallback(async () => {
        if (!dirHandle || !uploadCtx) return
        let queued = 0
        let uploading = 0
        try {
            const entries = await listFilesRecursively(dirHandle as any, ['.mp4'])
            for (const entry of entries) {
                if (!isMp4Name(entry.name)) continue
                const file = await entry.getFile()
                if (!isMp4File(file)) continue
                const relPath = entry.relativePath
                const { sha256 } = await computeSha256(file)
                const already = await hasProcessedHash(sha256)
                if (already) continue

                queued += 1
                uploading += 1
                setUploading(v => v + 1)
                setQueued(v => v + 1)

                try {
                    await addProcessedHash(sha256)
                    await uploadVideoFile(file, uploadCtx, undefined, relPath)
                    onUploaded()
                } catch (e: any) {
                    console.error('Upload failed for', relPath, e)
                    setLastError(e?.message || String(e))
                    await removeProcessedHash(sha256)
                } finally {
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
    }, [dirHandle, uploadCtx])

    React.useEffect(() => {
        if (timerRef.current) window.clearInterval(timerRef.current)
        if (!dirHandle || !uploadCtx) {
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
    }, [dirHandle, uploadCtx, intervalMs, scanOnce])

    return { active, lastRunAt, lastError, queued, uploading }
}
