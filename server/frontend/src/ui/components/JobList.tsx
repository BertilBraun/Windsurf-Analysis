import React from 'react'
import { JobSummary, JobStatus } from '../types'
import { AnimatedDots } from './AnimatedDots'
import JobThumbnail from './JobThumbnail'

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
            } else if (sortKey === 'name') {
                // Sort by local path when available
                const an = a.local_relative_path?.toLowerCase() ?? 'n/a'
                const bn = b.local_relative_path?.toLowerCase() ?? 'n/a'
                cmp = an < bn ? -1 : an > bn ? 1 : 0
            } else {
                throw new Error(`Unknown sort key: ${sortKey}`)
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
