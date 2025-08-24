import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { useJobs } from '../hooks/useJobs'
import { JobDetail, ReportType } from '../types'
import { JobList } from '../components/JobList'
import { IngressPanel } from '../components/IngressPanel'
import { loadDirectoryHandle, saveDirectoryHandle } from '../utils/idb'
import { CanvasPlayer } from '../player/CanvasPlayer'
import { Modal } from '../components/Modal'
import { KeyboardShortcutsModal } from '../components/KeyboardShortcutsModal'
import { Button } from '../components/Button'
import { SettingsModal } from '../components/SettingsModal'
import { PlayerModal } from '../components/PlayerModal'

export const MainPage: React.FC = () => {
    const { logout, email, authorizedFetch, authHeader } = useAuth()
    const { jobs, isPolling, startPolling, stopPolling, refreshJobDetail, deleteJob, reportJob } = useJobs()
    const [selectedJob, setSelectedJob] = React.useState<JobDetail | null>(null)
    const [showShortcuts, setShowShortcuts] = React.useState<boolean>(false)
    const [showSettings, setShowSettings] = React.useState<boolean>(false)

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

    return (
        <div>
            <div className="flex justify-between items-center mb-4">
                <div>
                    <strong>Welcome to Windsurf Analysis</strong>
                    {email ? ` ${email}` : ''}
                </div>
                <Button onClick={() => setShowSettings(true)} text="Settings" />
            </div>

            <IngressPanel
                dirHandle={dirHandle}
                dirPermission={dirPermission}
                onPickDirectory={pickDirectory}
                uploadCtx={uploadCtx}
                onUploaded={() => startPolling()}
            />

            <h3>Analyzed Videos</h3>
            <JobList jobs={jobs} onOpen={onOpen} openingId={openingId} dirHandle={dirHandle} />

            {selectedJob && (
                <PlayerModal
                    job={selectedJob}
                    dirHandle={dirHandle}
                    onClose={() => setSelectedJob(null)}
                    onDelete={onDelete}
                    onReport={onReport}
                    deletingId={deletingId}
                />
            )}
            {showShortcuts && <KeyboardShortcutsModal onClose={() => setShowShortcuts(false)} />}
            {showSettings && <SettingsModal onClose={() => setShowSettings(false)} onLogout={logout} />}
        </div>
    )
}
