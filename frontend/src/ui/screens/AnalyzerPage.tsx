import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { useJobs } from '../hooks/useJobs'
import { JobDetail, ReportType } from '../types'
import { JobList } from '../components/JobList'
import { IngressWidget } from '../components/IngressWidget'
import { loadDirectoryHandle, saveDirectoryHandle } from '../utils/idb'
import { KeyboardShortcutsModal } from '../components/KeyboardShortcutsModal'
import { Button } from '../components/Button'
import { SettingsModal } from '../components/SettingsModal'
import { PlayerModal } from '../components/PlayerModal'
import { LogoButton } from '../components/LogoButton'

export const AnalyzerPage: React.FC<{ onGoHome: () => void; onGoPricing: () => void }> = ({
    onGoHome,
    onGoPricing,
}) => {
    const { logout, user, authorizedFetch, getAuthHeader } = useAuth()
    const { jobs, refreshJobDetail, deleteJob, reportJob } = useJobs()
    const [selectedJob, setSelectedJob] = React.useState<JobDetail | null>(null)
    const [showShortcuts, setShowShortcuts] = React.useState<boolean>(false)
    const [showSettings, setShowSettings] = React.useState<boolean>(false)

    // Ingress folder handle stored at the page level for the whole session
    const [dirHandle, setDirHandle] = React.useState<FileSystemDirectoryHandle | null>(null)
    const [dirPermission, setDirPermission] = React.useState<'granted' | 'denied' | 'prompt' | null>(null)
    const uploadCtx = React.useMemo(() => ({ authorizedFetch, getAuthHeader }), [authorizedFetch, getAuthHeader])

    // Try to restore directory handle from IndexedDB on mount
    React.useEffect(() => {
        let cancelled = false
        ;(async () => {
            const stored = await loadDirectoryHandle()
            if (!stored || cancelled) return
            try {
                const p = await stored.queryPermission({ mode: 'read' })
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
            const handle = await window.showDirectoryPicker({ id: 'windsurf-ingress' })
            if (!handle) return
            const perm = await handle.requestPermission({ mode: 'read' })
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
            setSelectedJob(await refreshJobDetail(id))
        } finally {
            setOpeningId(null)
        }
    }

    let greeting = ''
    if (user) {
        if (user.displayName) greeting += ` — ${user.displayName}`
        else if (user.email) greeting += ` — ${user.email.split('@')[0]}`
    }

    return (
        <div className="min-h-dvh bg-white text-slate-900">
            <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                    <LogoButton onClick={onGoHome} />

                    <div className="text-sm text-slate-600">Upload, process, and review your jibes{greeting}.</div>

                    <div className="flex-1" />

                    <div className="flex items-center gap-2">
                        <Button size="sm" variant="ghost" onClick={() => setShowShortcuts(true)} text="Shortcuts" />
                        <Button size="sm" variant="secondary" onClick={() => setShowSettings(true)} text="Settings" />
                    </div>
                </div>
            </header>

            <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-6 flex flex-col gap-6">
                <div className="flex items-center justify-between">
                    <h2 className="m-0">Analyzed Videos</h2>
                    <div />
                </div>

                <JobList jobs={jobs} onOpen={onOpen} openingId={openingId} dirHandle={dirHandle} />

                <IngressWidget
                    dirHandle={dirHandle}
                    dirPermission={dirPermission}
                    onPickDirectory={pickDirectory}
                    uploadCtx={uploadCtx}
                    onUploaded={() => {}}
                />

                {selectedJob && (
                    <PlayerModal
                        job={selectedJob}
                        dirHandle={dirHandle}
                        onClose={() => setSelectedJob(null)}
                        onOpenNextJob={async () => {
                            if (!selectedJob || !jobs || jobs.length === 0) return
                            const idx = Math.max(
                                0,
                                jobs.findIndex(j => j.id === selectedJob.id)
                            )
                            const nextIdx = (idx + 1) % jobs.length
                            const target = jobs[nextIdx]
                            if (!target) return
                            const detail = await refreshJobDetail(target.id)
                            setSelectedJob(detail)
                        }}
                        onOpenPrevJob={async () => {
                            if (!selectedJob || !jobs || jobs.length === 0) return
                            const idx = Math.max(
                                0,
                                jobs.findIndex(j => j.id === selectedJob.id)
                            )
                            const prevIdx = (idx - 1 + jobs.length) % jobs.length
                            const target = jobs[prevIdx]
                            if (!target) return
                            const detail = await refreshJobDetail(target.id)
                            setSelectedJob(detail)
                        }}
                        onDelete={deleteJob}
                        onReport={reportJob}
                    />
                )}
                {showShortcuts && <KeyboardShortcutsModal onClose={() => setShowShortcuts(false)} />}
                {showSettings && <SettingsModal onClose={() => setShowSettings(false)} onLogout={logout} />}
            </main>

            {/* Beta badge: bottom-left, subtle (brand) */}
            <div className="fixed bottom-4 left-4">
                <button
                    type="button"
                    onClick={onGoPricing}
                    className="group flex items-center gap-2 rounded-full border border-brand-600/25 bg-white/90 backdrop-blur px-3 py-2 shadow-sm hover:shadow transition hover:cursor-pointer"
                    title="Beta — read more"
                    aria-label="Beta — read more"
                >
                    <span className="text-[11px] font-semibold text-brand-700 bg-red-400 text-white border border-brand-600/20 px-2 py-0.5 rounded-full">
                        Beta
                    </span>
                    <span className="text-xs text-slate-700 group-hover:text-slate-900">Read more</span>
                </button>
            </div>
        </div>
    )
}
