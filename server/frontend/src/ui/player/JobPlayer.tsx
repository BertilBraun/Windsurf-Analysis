import React from 'react'
import { JobDetail, ReportType } from '../types'

export const JobPlayer: React.FC<{
    job: JobDetail
    dirHandle: FileSystemDirectoryHandle | null
    onClose: () => void
    onDelete: (id: string) => void
    onReport: (id: string, type: ReportType, message: string) => void
}> = ({ job, dirHandle, onClose }) => {
    const [videoUrl, setVideoUrl] = React.useState<string | null>(null)
    const [error, setError] = React.useState<string | null>(null)

    const resolveFileFromRelativePath = React.useCallback(async () => {
        if (!dirHandle) {
            setError('No ingress folder selected')
            return null
        }
        try {
            const normalized = (job.original_file_path || '').replace(/^[./\\]+/, '')
            const parts = normalized.split(/[\\/]+/).filter(Boolean)
            let current: any = dirHandle
            for (let i = 0; i < parts.length; i++) {
                const name = parts[i]
                const isLast = i === parts.length - 1
                if (isLast) {
                    current = await current.getFileHandle(name)
                } else {
                    current = await current.getDirectoryHandle(name)
                }
            }
            const file = await (current as FileSystemFileHandle).getFile()
            return file
        } catch (e: any) {
            setError(e?.message || 'Failed to access file from folder')
            return null
        }
    }, [dirHandle, job.original_file_path])

    React.useEffect(() => {
        let revoked: string | null = null
        setVideoUrl(null)
        setError(null)
        ;(async () => {
            const file = await resolveFileFromRelativePath()
            if (!file) return
            const url = URL.createObjectURL(file)
            revoked = url
            setVideoUrl(url)
        })()
        return () => {
            if (revoked) URL.revokeObjectURL(revoked)
        }
    }, [resolveFileFromRelativePath])

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
                <video src={videoUrl} controls style={{ width: '100%', maxWidth: 960, background: '#000' }} />
            ) : (
                <div style={{ fontSize: 12, color: '#6b7280' }}>Loading video…</div>
            )}
        </div>
    )
}
