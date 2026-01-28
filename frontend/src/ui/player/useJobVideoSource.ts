/**
 * @fileoverview Hook for resolving video files from job metadata and file system sources.
 */

import React from 'react'
import { JobDetail } from '../types'
import { getFileByRelativePath } from '../utils/fsAccess'
import { VideoSource } from './videoSource'

/**
 * Error details for video source resolution failures.
 */
export type VideoSourceError = {
    /** Translation key for the error message. */
    key: string
    /** Technical error details, if available. */
    detail?: string
}

/**
 * Resolves a video file for a job based on the provided video source configuration.
 *
 * This hook handles both direct file sources and directory-based lookups. For directory
 * sources, it iterates through the job's `local_relative_paths` to find a matching file
 * within the provided directory handle.
 *
 * @param params - Hook parameters.
 * @param params.job - The job containing file path metadata.
 * @param params.videoSource - The source configuration (file or directory handle).
 * @param params.onFileLoaded - Callback invoked when a file is successfully resolved.
 * @returns An object containing the resolved `sourceFile`, a `fileMissing` flag, and any `error`.
 */
export function useJobVideoSource(params: {
    job: JobDetail
    videoSource: VideoSource
    onFileLoaded?: (file: File) => void
}) {
    const { job, videoSource, onFileLoaded } = params

    const [error, setError] = React.useState<VideoSourceError | null>(null)
    const [fileMissing, setFileMissing] = React.useState<boolean>(false)
    const [sourceFile, setSourceFile] = React.useState<File | null>(null)

    const onFileLoadedRef = React.useRef<typeof onFileLoaded>(onFileLoaded)
    React.useEffect(() => {
        onFileLoadedRef.current = onFileLoaded
    }, [onFileLoaded])

    const candidatesKey = React.useMemo(() => {
        const candidates = job.local_relative_paths
        if (!candidates || candidates.length === 0) return ''
        return candidates.join('\n')
    }, [job.local_relative_paths])

    const fileKey = React.useMemo(() => {
        if (videoSource.kind !== 'file') return ''
        const file = videoSource.file
        return `${file.name}|${file.type}|${file.size}|${file.lastModified}`
    }, [videoSource.kind === 'file' ? videoSource.file : null, videoSource.kind])

    const ingressDirHandle = videoSource.kind === 'ingress' ? videoSource.dirHandle : null

    React.useEffect(() => {
        let cancelled = false

        setSourceFile(null)
        setError(null)
        setFileMissing(false)

        const run = async () => {
            if (videoSource.kind === 'file') {
                setSourceFile(videoSource.file)
                onFileLoadedRef.current?.(videoSource.file)
                return
            }

            try {
                const dirHandle = videoSource.dirHandle
                if (!dirHandle) {
                    setError({ key: 'player.canvas.errors.noIngressFolder' })
                    return
                }

                const candidates = job.local_relative_paths
                if (!candidates) throw new Error('missing_local_paths')
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

                setSourceFile(file)
                onFileLoadedRef.current?.(file)
            } catch (e: any) {
                if (cancelled) return
                const msg = String(e?.message || '')
                const isMissing =
                    e?.name === 'NotFoundError' || /not\s*found|no such file|could not be found/i.test(msg)
                if (msg === 'missing_mapping' || msg === 'missing_local_paths') {
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
        }
    }, [job.id, candidatesKey, fileKey, ingressDirHandle, videoSource.kind])

    return { sourceFile, fileMissing, error }
}
