import React from 'react'
import { JobSummary, JobStatus } from '../types'
import { AnimatedDots } from './AnimatedDots'
import { Button } from './Button'

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
    if (jobs.length === 0) {
        return (
            <div className="text-center text-gray-500">
                Still looking for jobs
                <AnimatedDots />
            </div>
        )
    }
    return (
        <div>
            {jobs
                .sort((a, b) => (b.created_at < a.created_at ? -1 : 1))
                .map(j => (
                    <div key={j.id} className="flex items-center gap-2 p-2 border-b border-gray-300 hover:bg-gray-100">
                        <span className="font-mono text-sm">{j.original_file_path}</span>
                        <div className="flex-1" />
                        <StatusBadge status={j.status} />
                        <Button text="Open" isPending={openingId === j.id} onClick={() => onOpen(j.id)} />
                        <Button text="Delete" isPending={deletingId === j.id} onClick={() => onDelete(j.id)} />
                    </div>
                ))}
        </div>
    )
}
