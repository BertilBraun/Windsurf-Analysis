import React from 'react'
import { JobSummary } from '../types'
import { getFileByRelativePath } from '../utils/fsAccess'
import { getPathsForSha, getThumbnailBlob, saveThumbnailBlob } from '../utils/idb'
import { drawRotatedToCanvas } from '../player/rotation'

const THUMB_TARGET_W = 256
const THUMB_MIME = 'image/jpeg'
const THUMB_QUALITY = 0.7

function once(target: EventTarget, type: string, onError?: (e: any) => Error): Promise<void> {
    return new Promise((resolve, reject) => {
        const cleanup = () => {
            try {
                target.removeEventListener(type, onOk as any)
                if (onError) target.removeEventListener('error', onErr as any)
            } catch {}
        }
        const onOk = () => {
            cleanup()
            resolve()
        }
        const onErr = (e: any) => {
            cleanup()
            reject(onError ? onError(e) : new Error('event_error'))
        }
        target.addEventListener(type, onOk as any, { once: true } as any)
        if (onError) target.addEventListener('error', onErr as any, { once: true } as any)
    })
}

function canvasToBlob(canvas: HTMLCanvasElement, mime: string, quality: number): Promise<Blob> {
    return new Promise((resolve, reject) => {
        canvas.toBlob(
            blob => {
                if (!blob) reject(new Error('thumbnail_blob_null'))
                else resolve(blob)
            },
            mime,
            quality
        )
    })
}

async function resolveFirstExistingFile(
    dirHandle: FileSystemDirectoryHandle,
    candidates: string[]
): Promise<File | null> {
    for (const path of candidates) {
        try {
            return await getFileByRelativePath(dirHandle, path)
        } catch {
            // try next
        }
    }
    return null
}

async function generateThumbnailBlobFromVideo(file: File, dominantOrientation: number): Promise<Blob> {
    const videoUrl = URL.createObjectURL(file)
    try {
        const video = document.createElement('video')
        video.muted = true
        video.preload = 'metadata'
        video.src = videoUrl

        await once(video, 'loadedmetadata', () => new Error('video_error'))

        const seekTarget = Math.min(0.1, (video.duration || 1) - 0.1)
        try {
            video.currentTime = seekTarget
        } catch {}
        await once(video, 'seeked', () => new Error('video_error'))

        const oriented = document.createElement('canvas')
        drawRotatedToCanvas(video, oriented, dominantOrientation)

        const w = Math.max(1, oriented.width)
        const h = Math.max(1, oriented.height)
        const scale = Math.min(1, THUMB_TARGET_W / w)
        const outW = Math.max(1, Math.floor(w * scale))
        const outH = Math.max(1, Math.floor(h * scale))

        const canvas = document.createElement('canvas')
        canvas.width = outW
        canvas.height = outH
        const ctx = canvas.getContext('2d')!
        ctx.imageSmoothingEnabled = true
        ctx.imageSmoothingQuality = 'high'
        ctx.drawImage(oriented, 0, 0, outW, outH)
        const blob = await canvasToBlob(canvas, THUMB_MIME, THUMB_QUALITY)
        return blob
    } finally {
        URL.revokeObjectURL(videoUrl)
    }
}

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
    const lastCacheKeyRef = React.useRef<string>('')

    const cacheKey = React.useMemo(() => {
        const shaLower = String(job.original_checksum_sha256 || '').toLowerCase()
        const base = shaLower || `jobid:${job.id}`
        return `${base}:ori:${Number(job.dominant_orientation || 0)}:w${THUMB_TARGET_W}:q${Math.round(
            THUMB_QUALITY * 100
        )}`
    }, [job.id, job.original_checksum_sha256, job.dominant_orientation])

    React.useEffect(() => {
        let revokedThumbUrl: string | null = null
        let cancelled = false

        // Only reset the UI when the thumbnail identity changes.
        if (lastCacheKeyRef.current !== cacheKey) {
            lastCacheKeyRef.current = cacheKey
            setThumbUrl(null)
            setNotFound(false)
        }

        if (job.status !== 'succeeded') return
        if (!dirHandle) {
            setNotFound(true)
            return
        }

        ;(async () => {
            try {
                const cached = await getThumbnailBlob(cacheKey)
                if (cancelled) return

                if (cached) {
                    const url = URL.createObjectURL(cached)
                    revokedThumbUrl = url
                    setThumbUrl(url)
                    return
                }

                const candidates: string[] = []
                if (job.local_relative_path) candidates.push(job.local_relative_path)
                if (job.original_checksum_sha256) {
                    const extra = await getPathsForSha(String(job.original_checksum_sha256).toLowerCase())
                    for (const p of extra) if (!candidates.includes(p)) candidates.push(p)
                }
                if (candidates.length === 0) {
                    setNotFound(true)
                    return
                }

                const file = await resolveFirstExistingFile(dirHandle, candidates)
                if (cancelled) return

                if (!file) {
                    setNotFound(true)
                    return
                }

                const blob = await generateThumbnailBlobFromVideo(file, job.dominant_orientation)
                if (cancelled) return

                await saveThumbnailBlob(cacheKey, blob)

                const url = URL.createObjectURL(blob)
                revokedThumbUrl = url
                setThumbUrl(url)
            } catch (e: any) {
                if (!cancelled) setNotFound(true)
            }
        })()

        return () => {
            cancelled = true
            if (revokedThumbUrl) URL.revokeObjectURL(revokedThumbUrl)
        }
    }, [
        cacheKey,
        dirHandle,
        job.id,
        job.status,
        job.local_relative_path,
        job.original_checksum_sha256,
        job.dominant_orientation,
    ])

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
