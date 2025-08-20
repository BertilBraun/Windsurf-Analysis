import React from 'react'
import { JobSummary, JobStatus } from '../types'

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
    onDelete: (id: string) => void
    deletingId?: string | null
    openingId?: string | null
}> = ({ jobs, onOpen, onDelete, deletingId, openingId }) => {
    return (
        <div>
            {jobs.map(j => (
                <div key={j.id} className="flex items-center gap-2 p-2 border-b border-gray-300 hover:bg-gray-100">
                    <span className="font-mono text-sm">{j.original_file_path}</span>
                    <div className="flex-1" />
                    <StatusBadge status={j.status} />
                    <button onClick={() => onOpen(j.id)} disabled={openingId === j.id}>
                        {openingId === j.id ? 'Opening…' : 'Open'}
                    </button>
                    <button onClick={() => onDelete(j.id)} disabled={deletingId === j.id}>
                        {deletingId === j.id ? 'Deleting…' : 'Delete'}
                    </button>
                </div>
            ))}
        </div>
    )
}
