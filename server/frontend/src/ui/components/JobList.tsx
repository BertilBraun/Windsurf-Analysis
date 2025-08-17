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
        <span style={{ background: color, color: 'white', borderRadius: 12, padding: '2px 8px', fontSize: 12 }}>
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
                <div
                    key={j.id}
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        padding: 8,
                        borderBottom: '1px solid #eee',
                    }}
                >
                    <span style={{ width: 160, fontFamily: 'monospace' }}>{j.video_id.slice(0, 8)}</span>
                    <StatusBadge status={j.status} />
                    <span style={{ flex: 1 }}>{j.video_id}</span>
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
