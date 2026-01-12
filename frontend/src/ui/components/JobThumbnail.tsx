import React from 'react'
import { useTranslation } from 'react-i18next'
import { ALL_FORMATS, BlobSource, CanvasSink, Input } from 'mediabunny'
import { JobSummary } from '../types'
import { getFileByRelativePath } from '../utils/fsAccess'
import { getThumbnailBlob, saveThumbnailBlob } from '../utils/idb'
import { quantizeOrientation } from '../player/rotation'

const THUMB_TARGET_W = 256
const THUMB_MIME = 'image/jpeg'
const THUMB_QUALITY = 0.7

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

async function canvasLikeToBlob(
    canvas: HTMLCanvasElement | OffscreenCanvas,
    mime: string,
    quality: number
): Promise<Blob> {
    if (typeof (canvas as any).convertToBlob === 'function') {
        return (canvas as OffscreenCanvas).convertToBlob({ type: mime, quality } as any)
    }
    return canvasToBlob(canvas as HTMLCanvasElement, mime, quality)
}

async function hasReadPermission(dirHandle: FileSystemDirectoryHandle): Promise<boolean> {
    const dh: any = dirHandle as any
    if (typeof dh.queryPermission !== 'function') return true
    try {
        return (await dh.queryPermission({ mode: 'read' })) === 'granted'
    } catch {
        return false
    }
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
    const input = new Input({
        formats: ALL_FORMATS,
        source: new BlobSource(file),
    })

    try {
        const videoTrack = await input.getPrimaryVideoTrack()
        if (!videoTrack) throw new Error('no_primary_video_track')

        const startTimestamp = await videoTrack.getFirstTimestamp()
        const endTimestamp = await videoTrack.computeDuration()

        const safeStart = Math.max(0, startTimestamp)
        const safeEnd = Math.max(safeStart, endTimestamp)
        const duration = safeEnd - safeStart

        const timestamp = safeStart + Math.min(0.1, Math.max(0, duration))

        const sink = new CanvasSink(videoTrack, {
            width: THUMB_TARGET_W,
            rotation: quantizeOrientation(dominantOrientation),
        })

        const wrapped = await sink.getCanvas(timestamp)
        if (!wrapped) throw new Error('thumbnail_frame_null')

        return canvasLikeToBlob(wrapped.canvas as any, THUMB_MIME, THUMB_QUALITY)
    } finally {
        input.dispose()
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
    playable?: boolean
}> = ({ job, dirHandle, playable = true }) => {
    const { t } = useTranslation()
    const [thumbUrl, setThumbUrl] = React.useState<string | null>(null)
    const [notFound, setNotFound] = React.useState<boolean>(false)
    const lastCacheKeyRef = React.useRef<string>('')

    const cacheKey = React.useMemo(() => {
        const sha = String(job.sha256 || '')
        const base = sha || `jobid:${job.id}`
        return `${base}:ori:${Number(job.dominant_orientation || 0)}:w${THUMB_TARGET_W}:q${Math.round(
            THUMB_QUALITY * 100
        )}`
    }, [job.id, job.sha256, job.dominant_orientation])

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
        ;(async () => {
            if (!job.local_relative_paths) throw new Error('missing_local_paths')
            try {
                const cached = await getThumbnailBlob(cacheKey)
                if (cancelled) return

                if (cached) {
                    const url = URL.createObjectURL(cached)
                    revokedThumbUrl = url
                    setNotFound(false)
                    setThumbUrl(url)
                    return
                }

                if (!dirHandle) return

                setNotFound(false)

                const canRead = await hasReadPermission(dirHandle)
                if (cancelled || !canRead) return

                const candidates = job.local_relative_paths
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
                setNotFound(false)
                setThumbUrl(url)
            } catch (e: any) {
                if (!cancelled) setNotFound(false)
            }
        })()

        return () => {
            cancelled = true
            if (revokedThumbUrl) URL.revokeObjectURL(revokedThumbUrl)
        }
    }, [cacheKey, dirHandle, job.id, job.status, job.local_relative_paths, job.dominant_orientation])

    const boxClasses = 'relative w-48 h-28 bg-gray-200 rounded-md overflow-hidden flex items-center justify-center'

    if (notFound) {
        return (
            <div className={boxClasses}>
                <div className="text-red-600 font-bold text-center px-2">
                    {t('components.jobThumbnail.fileNotFound')}
                </div>
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
                : job.status === 'uploading'
                ? '#f59e0b'
                : job.status === 'canceled'
                ? '#9ca3af'
                : '#10b981'

        const statusKey = `components.jobThumbnail.status.${job.status}`
        const tooltipKey = `components.jobThumbnail.statusTooltip.${job.status}`

        return (
            <div className={boxClasses}>
                <span
                    className="text-white rounded-md px-2 py-1 text-sm text-center"
                    style={{ background: color }}
                    title={t(tooltipKey)}
                >
                    {t(statusKey)}
                </span>
            </div>
        )
    }

    return (
        <div className={boxClasses} title={t('components.jobThumbnail.openAnalysisTooltip')}>
            {thumbUrl ? (
                <>
                    <img
                        src={thumbUrl}
                        alt={job.local_relative_path ?? job.sha256}
                        className="absolute inset-0 w-full h-full object-cover"
                    />
                    {playable && <PlayOverlay />}
                </>
            ) : (
                <div className="text-gray-500 text-sm text-center">{t('components.jobThumbnail.generating')}</div>
            )}
        </div>
    )
}

export default JobThumbnail
