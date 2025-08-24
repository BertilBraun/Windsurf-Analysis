import React from 'react'
import { JobSummary, JobStatus } from '../types'
import { AnimatedDots } from './AnimatedDots'
import { getFileByRelativePath } from '../utils/fsAccess'

export const StatusBadge: React.FC<{ status: JobStatus }> = ({ status }) => {
    const color =
        status === 'succeeded'
            ? '#10b981'
            : status === 'failed'
            ? '#ef4444'
            : status === 'running'
            ? '#3b82f6'
            : status === 'pending'
            ? '#f59e0b'
            : '#9ca3af'
    return (
        <span className="text-white rounded-md px-2 py-1 text-sm" style={{ background: color }}>
            {status}
        </span>
    )
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

const JobThumbnail: React.FC<{
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
                        // Seek a tiny bit to get a decodable frame
                        video.currentTime = Math.min(0.1, (video.duration || 1) - 0.1)
                    } catch {}
                }
                const onSeeked = () => {
                    try {
                        const canvas = document.createElement('canvas')
                        const w = Math.max(1, Math.floor(video.videoWidth))
                        const h = Math.max(1, Math.floor(video.videoHeight))
                        const targetW = 256
                        const scale = Math.min(1, targetW / w)
                        canvas.width = Math.max(1, Math.floor(w * scale))
                        canvas.height = Math.max(1, Math.floor(h * scale))
                        const ctx = canvas.getContext('2d')!
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
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
    }, [job.id, job.status, job.local_relative_path, dirHandle])

    const boxClasses = 'relative w-48 h-28 bg-gray-200 rounded-md overflow-hidden flex items-center justify-center'

    if (job.status !== 'succeeded') {
        return (
            <div className={boxClasses}>
                <StatusBadge status={job.status} />
            </div>
        )
    }

    if (notFound) {
        return (
            <div className={boxClasses}>
                <div className="text-red-600 font-bold text-center px-2">VIDEO FILE NOT FOUND</div>
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
                <div className="text-gray-500 text-sm flex items-center gap-1">
                    Generating thumbnail
                    <AnimatedDots />
                </div>
            )}
        </div>
    )
}

export const JobList: React.FC<{
    jobs: JobSummary[]
    onOpen: (id: string) => void
    openingId?: string | null
    dirHandle?: FileSystemDirectoryHandle | null
}> = ({ jobs, onOpen, openingId, dirHandle = null }) => {
    const [sortKey, setSortKey] = React.useState<'name' | 'date'>('date')
    const [sortDir, setSortDir] = React.useState<'asc' | 'desc'>('desc')

    const toggleSort = (key: 'name' | 'date') => {
        if (key === sortKey) {
            setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
        } else {
            setSortKey(key)
            setSortDir(key === 'name' ? 'asc' : 'desc')
        }
    }

    const sortedJobs = React.useMemo(() => {
        const list = [...jobs]
        list.sort((a, b) => {
            let cmp = 0
            if (sortKey === 'date') {
                // Compare ISO-like timestamps; fallback to string compare
                cmp = a.created_at < b.created_at ? -1 : a.created_at > b.created_at ? 1 : 0
            } else {
                // Sort by local path when available
                const an = a.local_relative_path?.toLowerCase() ?? 'n/a'
                const bn = b.local_relative_path?.toLowerCase() ?? 'n/a'
                cmp = an < bn ? -1 : an > bn ? 1 : 0
            }
            return sortDir === 'asc' ? cmp : -cmp
        })
        return list
    }, [jobs, sortKey, sortDir])

    if (jobs.length === 0) {
        return (
            <div className="text-center text-gray-500">
                Still looking for jobs
                <AnimatedDots />
            </div>
        )
    }
    return (
        <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
                <div className="text-sm text-gray-600">Sort by</div>
                <div className="flex gap-2">
                    <button
                        className={`px-2 py-1 rounded-md text-sm border ${
                            sortKey === 'name'
                                ? 'bg-gray-700 text-gray-100 border-gray-700'
                                : 'bg-gray-100 text-gray-800 border-gray-300'
                        }`}
                        onClick={() => toggleSort('name')}
                    >
                        Name {sortKey === 'name' ? (sortDir === 'asc' ? '▲' : '▼') : '↕'}
                    </button>
                    <button
                        className={`px-2 py-1 rounded-md text-sm border ${
                            sortKey === 'date'
                                ? 'bg-gray-700 text-gray-100 border-gray-700'
                                : 'bg-gray-100 text-gray-800 border-gray-300'
                        }`}
                        onClick={() => toggleSort('date')}
                    >
                        Date {sortKey === 'date' ? (sortDir === 'asc' ? '▲' : '▼') : '↕'}
                    </button>
                </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {sortedJobs.map(j => {
                    const caption = j.local_relative_path?.replace(/\.mp4$/i, '') ?? 'n/a'
                    return (
                        <div key={j.id} className="flex flex-col items-start">
                            <div
                                className={`relative cursor-pointer ${openingId === j.id ? 'opacity-60' : ''}`}
                                onClick={() => onOpen(j.id)}
                                role="button"
                                tabIndex={0}
                                onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && onOpen(j.id)}
                            >
                                <JobThumbnail job={j} dirHandle={dirHandle} />
                                {openingId === j.id && (
                                    <div className="absolute inset-0 flex items-center justify-center">
                                        <div className="text-white text-sm bg-black/60 rounded px-2 py-1">Opening…</div>
                                    </div>
                                )}
                            </div>
                            <div className="mt-1 max-w-[12rem] truncate text-xs text-gray-700" title={caption}>
                                {caption}
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
