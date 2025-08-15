import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { useJobs } from '../hooks/useJobs'
import { JobDetail, ReportType } from '../types'
import { JobList } from '../components/JobList'
import { UploadControls } from '../components/UploadControls'
import { JobPlayer } from '../player/JobPlayer'

export const MainPage: React.FC<{ onLogout: () => void }> = ({ onLogout }) => {
    const { authorizedFetch, logout, email } = useAuth()
    const { jobs, isPolling, startPolling, stopPolling, refreshJobDetail, setJobs } = useJobs()
    const [selectedJob, setSelectedJob] = React.useState<JobDetail | null>(null)

    React.useEffect(() => {
        // Initial fetch but don't keep polling until needed
        startPolling()
        // immediately stop if there are no open jobs after first tick handled internally
        // stop when leaving page
        return () => stopPolling()
    }, [startPolling, stopPolling])

    const onOpen = async (id: string) => {
        const detail = await refreshJobDetail(id)
        setSelectedJob(detail)
    }

    const [deletingId, setDeletingId] = React.useState<string | null>(null)
    const onDelete = async (id: string) => {
        setDeletingId(id)
        try {
            await authorizedFetch(`/jobs/${id}`, { method: 'DELETE' })
            startPolling()
        } finally {
            setDeletingId(null)
        }
    }

    const onReport = async (id: string) => {
        const msg = window.prompt('Describe the issue')
        if (!msg) return
        const type: ReportType = 'other'
        await authorizedFetch(`/jobs/${id}/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type, message: msg }),
        })
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
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <div>
                    <strong>Welcome</strong>{email ? `, ${email}` : ''}
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <button onClick={isPolling ? stopPolling : startPolling}>{isPolling ? 'Stop polling' : 'Start polling'}</button>
                    <button onClick={handleLogout}>Logout</button>
                </div>
            </div>

            <div style={{ marginBottom: 16 }}>
                <UploadControls onSubmitted={onSubmitted} />
            </div>

            <h3>Jobs</h3>
            <JobList jobs={jobs} onOpen={onOpen} onDelete={onDelete} onReport={onReport} deletingId={deletingId} />

            {selectedJob && (
                <section style={{ marginTop: 16 }}>
                    <h3>Player</h3>
                    <JobPlayer job={selectedJob} onClose={() => setSelectedJob(null)} onDeleted={() => startPolling()} />
                </section>
            )}
        </div>
    )
}


