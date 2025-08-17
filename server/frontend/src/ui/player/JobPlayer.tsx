import React from 'react'
import { JobDetail, ReportType } from '../types'
import { getFileByRelativePath } from '../utils/fsAccess'

export const JobPlayer: React.FC<{
    job: JobDetail
    dirHandle: FileSystemDirectoryHandle | null
    onClose: () => void
    onDelete: (id: string) => void
    onReport: (id: string, type: ReportType, message: string) => void
}> = ({ job, dirHandle, onClose }) => {
    const [videoUrl, setVideoUrl] = React.useState<string | null>(null)
    const [error, setError] = React.useState<string | null>(null)
    const videoRef = React.useRef<HTMLVideoElement | null>(null)
    const fallbackTriedRef = React.useRef(false)

    const log = React.useCallback((...args: any[]) => {
        // Prefix logs for easy filtering
        // eslint-disable-next-line no-console
        console.log('[JobPlayer]', ...args)
    }, [])

    const resolveFileFromRelativePath = React.useCallback(async () => {
        if (!dirHandle) {
            setError('No ingress folder selected')
            log('No dirHandle; cannot resolve', job.original_file_path)
            return null
        }
        try {
            const file = await getFileByRelativePath(dirHandle, job.original_file_path)
            log('Resolved file', { name: file.name, size: file.size, type: file.type })
            return file
        } catch (e: any) {
            setError(e?.message || 'Failed to access file from folder')
            log('Error resolving file', e)
            return null
        }
    }, [dirHandle, job.original_file_path, log])

    React.useEffect(() => {
        let revoked: string | null = null
        setVideoUrl(null)
        setError(null)
        ;(async () => {
            log('Begin resolve for job', job.id)
            const file = await resolveFileFromRelativePath()
            if (!file) return
            const url = URL.createObjectURL(file)
            log('Created object URL', url)
            revoked = url
            setVideoUrl(url)
        })()
        return () => {
            if (revoked) {
                log('Revoking object URL', revoked)
                URL.revokeObjectURL(revoked)
            }
        }
    }, [resolveFileFromRelativePath, job.id, log])

    React.useEffect(() => {
        const v = videoRef.current
        if (!v) return
        const onLoadedMetadata = () =>
            log('video loadedmetadata', { duration: v.duration, videoWidth: v.videoWidth, videoHeight: v.videoHeight })
        const onLoadedData = () => log('video loadeddata')
        const onCanPlay = () => log('video canplay')
        const onCanPlayThrough = () => log('video canplaythrough')
        const onError = async () => {
            log('video error', v.error)
            if (!fallbackTriedRef.current) {
                fallbackTriedRef.current = true
                // For now, do not transcode automatically; show a simple error instead
                setError('This file cannot be played in your browser. Please use MP4 (H.264).')
            }
        }
        const onStalled = () => log('video stalled')
        const onWaiting = () => log('video waiting')
        const onPlay = () => log('video play')
        const onPause = () => log('video pause')
        const onEnded = () => log('video ended')
        v.addEventListener('loadedmetadata', onLoadedMetadata)
        v.addEventListener('loadeddata', onLoadedData)
        v.addEventListener('canplay', onCanPlay)
        v.addEventListener('canplaythrough', onCanPlayThrough)
        v.addEventListener('error', onError)
        v.addEventListener('stalled', onStalled)
        v.addEventListener('waiting', onWaiting)
        v.addEventListener('play', onPlay)
        v.addEventListener('pause', onPause)
        v.addEventListener('ended', onEnded)
        return () => {
            v.removeEventListener('loadedmetadata', onLoadedMetadata)
            v.removeEventListener('loadeddata', onLoadedData)
            v.removeEventListener('canplay', onCanPlay)
            v.removeEventListener('canplaythrough', onCanPlayThrough)
            v.removeEventListener('error', onError)
            v.removeEventListener('stalled', onStalled)
            v.removeEventListener('waiting', onWaiting)
            v.removeEventListener('play', onPlay)
            v.removeEventListener('pause', onPause)
            v.removeEventListener('ended', onEnded)
        }
    }, [videoUrl, log])

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <strong>Job: {job.id}</strong>
                <button onClick={onClose}>Close</button>
            </div>
            <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 8 }}>
                Local path: {job.original_file_path || '(unknown)'}
            </div>
            {error && <div style={{ color: '#b91c1c', fontSize: 12, marginBottom: 8 }}>{error}</div>}
            {videoUrl ? (
                <video
                    ref={videoRef}
                    key={videoUrl}
                    src={videoUrl}
                    controls
                    playsInline
                    muted={false}
                    preload="metadata"
                    style={{ width: '100%', maxWidth: 960, background: '#000' }}
                />
            ) : (
                <div style={{ fontSize: 12, color: '#6b7280' }}>Loading video…</div>
            )}
        </div>
    )
}
