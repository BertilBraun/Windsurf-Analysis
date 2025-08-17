import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { useJobs } from '../hooks/useJobs'
import { JobDetail, ReportType } from '../types'
import { JobList } from '../components/JobList'
import { UploadControls } from '../components/UploadControls'
import { JobPlayer } from '../player/JobPlayer'
import { useIngressScanner } from '../hooks/useIngressScanner'
import { loadDirectoryHandle, saveDirectoryHandle } from '../utils/idb'

export const MainPage: React.FC<{ onLogout: () => void }> = ({ onLogout }) => {
    const { logout, email, authorizedFetch, authHeader } = useAuth()
    const { jobs, isPolling, startPolling, stopPolling, refreshJobDetail, deleteJob, reportJob } = useJobs()
    const [selectedJob, setSelectedJob] = React.useState<JobDetail | null>(null)

    React.useEffect(() => {
        // Initial fetch but don't keep polling until needed
        startPolling()
        // immediately stop if there are no open jobs after first tick handled internally
        // stop when leaving page
        return () => stopPolling()
    }, [startPolling, stopPolling])

    // Ingress folder handle stored at the page level for the whole session
    const [dirHandle, setDirHandle] = React.useState<FileSystemDirectoryHandle | null>(null)
    const [dirPermission, setDirPermission] = React.useState<'granted' | 'denied' | 'prompt' | null>(null)
    const uploadCtx = React.useMemo(() => ({ authorizedFetch, authHeader }), [authorizedFetch, authHeader])
    const scanner = useIngressScanner(dirHandle, uploadCtx)

    // Try to restore directory handle from IndexedDB on mount
    React.useEffect(() => {
        let cancelled = false
        ;(async () => {
            const stored = await loadDirectoryHandle()
            if (!stored || cancelled) return
            try {
                const p = await (stored as any).queryPermission?.({ mode: 'read' })
                setDirPermission(p || null)
                if (p === 'granted') setDirHandle(stored)
            } catch {}
        })()
        return () => {
            cancelled = true
        }
    }, [])

    const pickDirectory = async () => {
        try {
            const handle = await (window as any).showDirectoryPicker?.({ id: 'windsurf-ingress' })
            if (!handle) return
            const perm = await handle.requestPermission?.({ mode: 'read' })
            setDirPermission(perm || null)
            setDirHandle(handle)
            await saveDirectoryHandle(handle)
        } catch (e) {
            // user cancelled or unsupported
        }
    }

    const [openingId, setOpeningId] = React.useState<string | null>(null)
    const onOpen = async (id: string) => {
        setOpeningId(id)
        try {
            const detail = await refreshJobDetail(id)
            setSelectedJob(detail)
        } finally {
            setOpeningId(null)
        }
    }

    const [deletingId, setDeletingId] = React.useState<string | null>(null)
    const onDelete = async (id: string) => {
        setDeletingId(id)
        try {
            await deleteJob(id)
        } finally {
            setDeletingId(null)
        }
    }

    const onReport = async (id: string, type: ReportType, message: string) => {
        await reportJob(id, type, message)
        startPolling()
    }

    const onSubmitted = (num: number) => {
        startPolling()
    }

    const handleLogout = () => {
        logout()
        onLogout()
    }

    return (
        <div>
            {/* Prominent ingress folder selector */}
            <div
                style={{
                    padding: 12,
                    border: '1px solid #ddd',
                    borderRadius: 8,
                    marginBottom: 16,
                    background: '#f9fafb',
                }}
            >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <strong>Ingress folder</strong>
                        <span style={{ fontSize: 12, color: '#6b7280' }}>
                            Select your "windsurf analysis videos" folder. We'll monitor it every 10s and auto-upload
                            new videos.
                        </span>
                        {dirHandle ? (
                            <span style={{ fontSize: 12, color: '#374151', marginTop: 4 }}>
                                Selected: {(dirHandle as any).name || 'Folder'}{' '}
                                {dirPermission !== 'granted' ? '(permission pending)' : ''}
                            </span>
                        ) : (
                            <span style={{ fontSize: 12, color: '#b91c1c', marginTop: 4 }}>No folder selected</span>
                        )}
                        {scanner.active && (
                            <span style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
                                Scanning... queued {scanner.queued}, uploading {scanner.uploading}
                                {scanner.lastRunAt
                                    ? ` (last run ${new Date(scanner.lastRunAt).toLocaleTimeString()})`
                                    : ''}
                                {scanner.lastError ? ` — ${scanner.lastError}` : ''}
                            </span>
                        )}
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <button onClick={pickDirectory}>{dirHandle ? 'Change folder' : 'Select folder'}</button>
                    </div>
                </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <div>
                    <strong>Welcome</strong>
                    {email ? `, ${email}` : ''}
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <button onClick={isPolling ? stopPolling : startPolling}>
                        {isPolling ? 'Stop polling' : 'Start polling'}
                    </button>
                    <button onClick={handleLogout}>Logout</button>
                </div>
            </div>

            <div style={{ marginBottom: 16 }}>
                <UploadControls onSubmitted={onSubmitted} />
            </div>

            <h3>Jobs</h3>
            <JobList jobs={jobs} onOpen={onOpen} onDelete={onDelete} deletingId={deletingId} openingId={openingId} />

            {selectedJob && (
                <section style={{ marginTop: 16 }}>
                    <h3>Player</h3>
                    <JobPlayer
                        job={selectedJob}
                        dirHandle={dirHandle}
                        onClose={() => setSelectedJob(null)}
                        onDelete={onDelete}
                        onReport={onReport}
                    />
                </section>
            )}
        </div>
    )
}
