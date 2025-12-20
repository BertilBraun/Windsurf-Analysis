import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { useAuth } from '../auth/AuthProvider'
import { useJobs } from '../hooks/useJobs'
import { JobDetail } from '../types'
import { JobList, getJobListOrderedJobIds, type JobListSortDir, type JobListSortKey } from '../components/JobList'
import { IngressWidget } from '../components/IngressWidget'
import { loadDirectoryHandle, loadSetting, saveDirectoryHandle, saveSetting } from '../utils/idb'
import { Button } from '../components/Button'
import { LanguageSwitcher } from '../components/LanguageSwitcher'
import { SettingsModal } from '../components/SettingsModal'
import { PlayerModal } from '../components/PlayerModal'
import { LogoButton } from '../components/LogoButton'
import { trackEvent } from '../utils/analytics'
import { AnalyzerTutorialModal } from '../components/AnalyzerTutorialModal'

const ANALYZER_TUTORIAL_SEEN_KEY = 'analyzerTutorialSeen.v1'

export const AnalyzerPage: React.FC<{ onGoHome: () => void; onGoPricing: () => void }> = ({
    onGoHome,
    onGoPricing,
}) => {
    const { t } = useTranslation()
    const { logout, user, authorizedFetch, getAuthHeader } = useAuth()
    const { jobs, initialSyncComplete: jobsInitialSyncComplete, refreshJobDetail, deleteJob, reportJob } = useJobs()
    const [selectedJob, setSelectedJob] = React.useState<JobDetail | null>(null)
    const [showSettings, setShowSettings] = React.useState<boolean>(false)
    const [showTutorial, setShowTutorial] = React.useState<boolean>(false)
    const [tutorialStepIndex, setTutorialStepIndex] = React.useState<number>(0)
    const [sortKey, setSortKey] = React.useState<JobListSortKey>('date')
    const [sortDir, setSortDir] = React.useState<JobListSortDir>('desc')

    // Workaround: some TS tooling instances may cache older prop typings during rapid edits.
    // This keeps runtime behavior correct while avoiding a stale "extra props" diagnostic.
    const TutorialModal = AnalyzerTutorialModal as React.FC<any>

    // Ingress folder handle stored at the page level for the whole session
    const [dirHandle, setDirHandle] = React.useState<FileSystemDirectoryHandle | null>(null)
    const [dirPermission, setDirPermission] = React.useState<'granted' | 'denied' | 'prompt' | null>(null)
    const uploadCtx = React.useMemo(() => ({ authorizedFetch, getAuthHeader }), [authorizedFetch, getAuthHeader])
    const knownChecksumsSha256 = React.useMemo(() => {
        const s = new Set<string>()
        for (const j of jobs) {
            const sha = String(j.original_checksum_sha256 || '').toLowerCase()
            if (sha) s.add(sha)
        }
        return s
    }, [jobs])

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

    // Auto-open the tutorial once for new users (no ingress folder and no jobs yet).
    React.useEffect(() => {
        loadSetting<boolean>(ANALYZER_TUTORIAL_SEEN_KEY).then(seen => {
            if (seen) return
            trackEvent('open_tutorial', { source: 'auto' })
            setShowTutorial(true)
        })
    }, [])

    const openTutorial = React.useCallback(async (source: 'header' | 'empty_state' | 'auto') => {
        trackEvent('open_tutorial', { source })
        setShowTutorial(true)
    }, [])

    const pickDirectory = async () => {
        try {
            const handle = await window.showDirectoryPicker({ id: 'windsurf-ingress' })
            if (!handle) return
            const perm = await handle.requestPermission({ mode: 'read' })
            setDirPermission(perm || null)
            setDirHandle(handle)
            await saveDirectoryHandle(handle)
            trackEvent('ingress_folder_selected', { permission: perm || null })
        } catch (e) {
            // user cancelled or unsupported
            const name = (e as any)?.name
            trackEvent('ingress_folder_select_failed', { name: name ? String(name) : 'unknown' })
        }
    }

    const [openingId, setOpeningId] = React.useState<string | null>(null)
    const onOpen = async (id: string, localRelativePath?: string | null) => {
        setOpeningId(id)
        try {
            trackEvent('job_open', { job_id: id })
            const detail = await refreshJobDetail(id)
            setSelectedJob(localRelativePath ? { ...detail, local_relative_path: localRelativePath } : detail)
        } finally {
            setOpeningId(null)
        }
    }

    const toggleSort = React.useCallback(
        (key: JobListSortKey) => {
            if (key === sortKey) {
                setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
            } else {
                setSortKey(key)
                setSortDir(key === 'name' ? 'asc' : 'desc')
            }
        },
        [sortKey]
    )

    const orderedSucceededJobIds = React.useMemo(() => {
        const succeeded = jobs.filter(j => j.status === 'succeeded')
        return getJobListOrderedJobIds(succeeded, sortKey, sortDir)
    }, [jobs, sortKey, sortDir])

    const [deletingId, setDeletingId] = React.useState<string | null>(null)
    const onDelete = async (id: string) => {
        setDeletingId(id)
        try {
            await deleteJob(id)
        } finally {
            setDeletingId(null)
            setSelectedJob(null)
        }
    }

    const userLabel = user?.displayName ?? (user?.email ? user.email.split('@')[0] : '')
    const headerText = userLabel
        ? t('screens.analyzer.header.taglineWithName', { name: userLabel })
        : t('screens.analyzer.header.tagline')

    return (
        <div className="min-h-dvh bg-white text-slate-900">
            <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                    <LogoButton onClick={onGoHome} />

                    <div className="text-sm text-slate-600">{headerText}</div>

                    <div className="flex-1" />

                    <div className="flex items-center gap-2">
                        <LanguageSwitcher />
                        <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => {
                                openTutorial('header')
                            }}
                            text={t('screens.analyzer.actions.tutorial')}
                        />
                        <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => {
                                trackEvent('open_settings')
                                setShowSettings(true)
                            }}
                            text={t('screens.analyzer.actions.settings')}
                        />
                    </div>
                </div>
            </header>

            <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-6 flex flex-col gap-6">
                {!dirHandle && (
                    <section className="rounded-2xl border border-slate-200 bg-slate-50 p-4 sm:p-5">
                        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
                            <div className="flex-1">
                                <div className="text-sm font-semibold text-slate-900">
                                    {t('screens.analyzer.emptyState.title')}
                                </div>
                                <div className="mt-1 text-sm text-slate-600">
                                    <Trans i18nKey="screens.analyzer.emptyState.body" components={{ b: <b /> }} />
                                </div>
                            </div>
                            <div className="flex flex-wrap gap-2">
                                <Button
                                    variant="secondary"
                                    onClick={() => {
                                        trackEvent('tutorial_select_folder_clicked')
                                        void pickDirectory()
                                    }}
                                    text={
                                        dirHandle ? t('common.actions.changeFolder') : t('common.actions.selectFolder')
                                    }
                                />
                                <Button
                                    variant="primary"
                                    onClick={() => openTutorial('empty_state')}
                                    text={t('screens.analyzer.emptyState.openTutorial')}
                                />
                            </div>
                        </div>
                    </section>
                )}

                <JobList
                    jobs={jobs}
                    sortKey={sortKey}
                    sortDir={sortDir}
                    onToggleSort={toggleSort}
                    onOpen={onOpen}
                    openingId={openingId}
                    dirHandle={dirHandle}
                />

                <IngressWidget
                    dirHandle={dirHandle}
                    dirPermission={dirPermission}
                    onPickDirectory={pickDirectory}
                    uploadCtx={uploadCtx}
                    knownChecksumsSha256={knownChecksumsSha256}
                    enabled={jobsInitialSyncComplete}
                />

                {selectedJob && (
                    <PlayerModal
                        job={selectedJob}
                        dirHandle={dirHandle}
                        onClose={() => setSelectedJob(null)}
                        onOpenNextJob={async () => {
                            if (!selectedJob || orderedSucceededJobIds.length === 0) return
                            const idx = Math.max(
                                0,
                                orderedSucceededJobIds.findIndex(id => id === selectedJob.id)
                            )
                            const nextIdx = (idx + 1) % orderedSucceededJobIds.length
                            const targetId = orderedSucceededJobIds[nextIdx]
                            if (!targetId) return
                            const detail = await refreshJobDetail(targetId)
                            setSelectedJob(detail)
                        }}
                        onOpenPrevJob={async () => {
                            if (!selectedJob || orderedSucceededJobIds.length === 0) return
                            const idx = Math.max(
                                0,
                                orderedSucceededJobIds.findIndex(id => id === selectedJob.id)
                            )
                            const prevIdx = (idx - 1 + orderedSucceededJobIds.length) % orderedSucceededJobIds.length
                            const targetId = orderedSucceededJobIds[prevIdx]
                            if (!targetId) return
                            const detail = await refreshJobDetail(targetId)
                            setSelectedJob(detail)
                        }}
                        onDelete={onDelete}
                        onReport={reportJob}
                        deletingId={deletingId}
                    />
                )}
                {showSettings && <SettingsModal onClose={() => setShowSettings(false)} onLogout={logout} />}
                {showTutorial && (
                    <TutorialModal
                        onClose={() => {
                            trackEvent('close_tutorial')
                            setShowTutorial(false)
                            void saveSetting(ANALYZER_TUTORIAL_SEEN_KEY, true)
                        }}
                        stepIndex={tutorialStepIndex}
                        onStepIndexChange={setTutorialStepIndex}
                        onPickIngressFolder={() => void pickDirectory()}
                        ingressFolderName={dirHandle?.name ?? null}
                    />
                )}
            </main>

            {/* Beta badge: bottom-left, subtle (brand) */}
            <div className="fixed bottom-4 left-4">
                <button
                    type="button"
                    onClick={() => {
                        trackEvent('open_pricing_from_beta_badge')
                        onGoPricing()
                    }}
                    className="group flex items-center gap-2 rounded-full border border-brand-600/25 bg-white/90 backdrop-blur px-3 py-2 shadow-sm hover:shadow transition hover:cursor-pointer"
                    title={t('screens.analyzer.beta.title')}
                >
                    <span className="text-[11px] font-semibold text-white bg-brand-600 border border-brand-600/20 px-2 py-0.5 rounded-full">
                        {t('screens.analyzer.beta.label')}
                    </span>
                    <span className="text-xs text-slate-700 group-hover:text-slate-900">
                        {t('screens.analyzer.beta.action')}
                    </span>
                </button>
            </div>
        </div>
    )
}
