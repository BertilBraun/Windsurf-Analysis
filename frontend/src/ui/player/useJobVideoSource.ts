import React from 'react'
import { JobDetail } from '../types'
import { getFileByRelativePath } from '../utils/fsAccess'
import { getPathsForSha } from '../utils/idb'

export type VideoSourceError = { key: string; detail?: string }

export function useJobVideoSource(params: {
    job: JobDetail
    dirHandle: FileSystemDirectoryHandle | null
    onFileLoaded?: (file: File) => void
}) {
    const { job, dirHandle, onFileLoaded } = params

    const [error, setError] = React.useState<VideoSourceError | null>(null)
    const [fileMissing, setFileMissing] = React.useState<boolean>(false)
    const [videoUrl, setVideoUrl] = React.useState<string | null>(null)
    const [sourceFile, setSourceFile] = React.useState<File | null>(null)

    React.useEffect(() => {
        let revoked: string | null = null
        let cancelled = false

        setVideoUrl(null)
        setSourceFile(null)
        setError(null)
        setFileMissing(false)

        const run = async () => {
            if (!dirHandle) {
                setError({ key: 'player.canvas.errors.noIngressFolder' })
                return
            }

            try {
                const candidates: string[] = []
                if (job.local_relative_path) candidates.push(job.local_relative_path)
                if (job.original_checksum_sha256) {
                    const extra = await getPathsForSha(String(job.original_checksum_sha256).toLowerCase())
                    for (const p of extra) if (!candidates.includes(p)) candidates.push(p)
                }
                if (candidates.length === 0) throw new Error('missing_mapping')

                let file: File | null = null
                for (const path of candidates) {
                    try {
                        file = await getFileByRelativePath(dirHandle, path)
                        break
                    } catch (e: any) {
                        const msg = String(e?.message || '')
                        const isMissing =
                            e?.name === 'NotFoundError' || /not\s*found|no such file|could not be found/i.test(msg)
                        if (isMissing) continue
                        throw e
                    }
                }

                if (!file) throw new Error('file_not_found')
                if (cancelled) return

                const url = URL.createObjectURL(file)
                revoked = url
                setVideoUrl(url)
                setSourceFile(file)
                onFileLoaded?.(file)
            } catch (e: any) {
                if (cancelled) return
                const msg = String(e?.message || '')
                const isMissing =
                    e?.name === 'NotFoundError' || /not\s*found|no such file|could not be found/i.test(msg)
                if (msg === 'missing_mapping') {
                    setError({ key: 'player.canvas.errors.missingMapping' })
                } else if (msg === 'file_not_found' || isMissing) {
                    setFileMissing(true)
                    setError({ key: 'player.canvas.errors.fileNotFound' })
                } else {
                    setError({ key: 'player.canvas.errors.accessFailed', detail: msg })
                }
            }
        }

        run()

        return () => {
            cancelled = true
            if (revoked) URL.revokeObjectURL(revoked)
        }
    }, [dirHandle, job.id, job.local_relative_path, job.original_checksum_sha256, onFileLoaded])

    return { videoUrl, sourceFile, fileMissing, error }
}
