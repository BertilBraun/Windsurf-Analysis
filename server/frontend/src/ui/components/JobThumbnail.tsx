import React from 'react'
import { JobSummary } from '../types'
import { getFileByRelativePath } from '../utils/fsAccess'
import { drawRotatedToCanvas } from '../player/rotation'

const PlayOverlay: React.FC = () => (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-12 h-12 rounded-full bg-black/60 flex items-center justify-center">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
                <path d="M8 5v14l11-7L8 5z" />
            </svg>
        </div>
    </div>
)

export const JobThumbnail: React.FC<{
    job: JobSummary
    dirHandle: FileSystemDirectoryHandle | null
}> = ({ job, dirHandle }) => {
    const [thumbUrl, setThumbUrl] = React.useState<string | null>(null)
    const [notFound, setNotFound] = React.useState<boolean>(false)

    React.useEffect(() => {
        let revoked: string | null = null
        let cancelled = false
        setThumbUrl(null)
        setNotFound(false)

        if (job.status !== 'succeeded') return
        if (!dirHandle) {
            setNotFound(true)
            return
        }

        ;(async () => {
            try {
                const path = job.local_relative_path
                if (!path) {
                    setNotFound(true)
                    return
                }
                const file = await getFileByRelativePath(dirHandle, path)
                if (!file) {
                    if (!cancelled) setNotFound(true)
                    return
                }
                const url = URL.createObjectURL(file)
                revoked = url
                const video = document.createElement('video')
                video.muted = true
                video.preload = 'metadata'
                video.src = url

                const captureFrame = async () => {
                    try {
                        video.currentTime = Math.min(0.1, (video.duration || 1) - 0.1)
                    } catch {}
                }

                const onSeeked = () => {
                    try {
                        // Draw rotated frame to offscreen then scale to target size
                        const oriented = document.createElement('canvas')
                        drawRotatedToCanvas(video, oriented, job.dominant_orientation)

                        const targetW = 256
                        const w = Math.max(1, oriented.width)
                        const h = Math.max(1, oriented.height)
                        const scale = Math.min(1, targetW / w)
                        const outW = Math.max(1, Math.floor(w * scale))
                        const outH = Math.max(1, Math.floor(h * scale))

                        const canvas = document.createElement('canvas')
                        canvas.width = outW
                        canvas.height = outH
                        const ctx = canvas.getContext('2d')!
                        ctx.imageSmoothingEnabled = true
                        ctx.imageSmoothingQuality = 'high'
                        ctx.drawImage(oriented, 0, 0, outW, outH)
                        const dataUrl = canvas.toDataURL('image/jpeg', 0.7)
                        if (!cancelled) setThumbUrl(dataUrl)
                    } catch {
                        if (!cancelled) setNotFound(true)
                    }
                }

                video.addEventListener('loadedmetadata', captureFrame, { once: true })
                video.addEventListener('seeked', onSeeked, { once: true })
                video.addEventListener('error', () => !cancelled && setNotFound(true), { once: true })
            } catch {
                if (!cancelled) setNotFound(true)
            }
        })()

        return () => {
            cancelled = true
            if (revoked) URL.revokeObjectURL(revoked)
        }
    }, [job.id, job.status, job.local_relative_path, job.dominant_orientation, dirHandle])

    const boxClasses = 'relative w-48 h-28 bg-gray-200 rounded-md overflow-hidden flex items-center justify-center'

    if (notFound) {
        return (
            <div className={boxClasses}>
                <div className="text-red-600 font-bold text-center px-2">VIDEO FILE NOT FOUND</div>
            </div>
        )
    }

    // For non-succeeded jobs, show a status badge inside the box instead of a loading text
    if (job.status !== 'succeeded') {
        const isProcessing =
            job.status === 'starting' ||
            job.status === 'orientation' ||
            job.status === 'stabilization' ||
            job.status === 'detection' ||
            job.status === 'appearance' ||
            job.status === 'tracking'
        const color =
            job.status === 'failed'
                ? '#ef4444'
                : isProcessing
                ? '#3b82f6'
                : job.status === 'pending'
                ? '#f59e0b'
                : job.status === 'canceled'
                ? '#9ca3af'
                : '#10b981'

        const text =
            job.status === 'canceled'
                ? 'Canceled'
                : job.status === 'failed'
                ? 'Failed'
                : job.status === 'starting'
                ? 'Starting'
                : job.status === 'orientation'
                ? 'Orienting Video'
                : job.status === 'stabilization'
                ? 'Stabilizing Video'
                : job.status === 'detection'
                ? 'Detecting Surfers'
                : job.status === 'appearance'
                ? 'Surfer Identification'
                : job.status === 'tracking'
                ? 'Tracking Surfers'
                : job.status === 'pending'
                ? 'Pending'
                : 'Succeeded'

        return (
            <div className={boxClasses}>
                <span className="text-white rounded-md px-2 py-1 text-sm" style={{ background: color }}>
                    {text}
                </span>
            </div>
        )
    }

    return (
        <div className={boxClasses}>
            {thumbUrl ? (
                <>
                    <img
                        src={thumbUrl}
                        alt={job.local_relative_path ?? job.original_checksum_sha256}
                        className="absolute inset-0 w-full h-full object-cover"
                    />
                    <PlayOverlay />
                </>
            ) : (
                <div className="text-gray-500 text-sm">Generating thumbnail…</div>
            )}
        </div>
    )
}

export default JobThumbnail
