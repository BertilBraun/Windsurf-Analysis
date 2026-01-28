/**
 * @module AnalyzerPage
 * Main dashboard for authenticated users to manage analysis jobs, ingress folders, and view results.
 */

import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { useAuth } from '../auth/AuthProvider'
import { useJobs } from '../hooks/useJobs'
import { JobDetail } from '../types'
import { JobList, getJobListOrderedJobIds, type JobListSortDir, type JobListSortKey } from '../components/JobList'
import { IngressWidget } from '../components/IngressWidget'
import { loadDirectoryHandle, saveDirectoryHandle } from '../utils/idb'
import { Button } from '../components/Button'
import { SettingsModal } from '../components/SettingsModal'
import { PlayerModal } from '../components/PlayerModal'
import { HelpModal } from '../components/HelpModal'
import { LogoButton } from '../components/LogoButton'
import { FeedbackModal } from '../components/FeedbackModal'
import { trackEvent } from '../utils/analytics'
import { AnalyzerTutorialModal } from '../components/AnalyzerTutorialModal'
import { useTutorialController } from '../hooks/useTutorialController'
import { useOnce } from '../hooks/useOnce'
import settingsIcon from '../assets/settings.svg'
import { assert } from '../utils/assert'
import { notifyIngressDirectoryChanged, subscribeIngressDirectoryChanged } from '../utils/ingressDirectorySync'
import { UnsupportedBrowserBanner } from '../components/UnsupportedBrowserBanner'

const FEEDBACK_PROMPT_SEEN_KEY = 'feedbackPromptSeen.v1'
const PLAYER_OPENED_ONCE_KEY = 'player.openedOnce.v1'

/**
 * The primary application screen for authenticated users to manage and view analysis jobs.
 *
 * @param props - Component properties.
 * @param props.onGoHome - Callback to navigate to the landing page.
 * @param props.onGoPricing - Callback to navigate to the pricing page.
 */
export const AnalyzerPage: React.FC<{ onGoHome: () => void; onGoPricing: () => void }> = ({
    onGoHome,
    onGoPricing,
}) => {
    const { t } = useTranslation()
    const { logout, user, authorizedFetch } = useAuth()
    const { jobs, initialSyncComplete: jobsInitialSyncComplete, refreshJobDetail, deleteJobs, reportJob } = useJobs()
    const [selectedJob, setSelectedJob] = React.useState<JobDetail | null>(null)
    const [showSettings, setShowSettings] = React.useState<boolean>(false)
    const [showHelp, setShowHelp] = React.useState<boolean>(false)
    const [sortKey, setSortKey] = React.useState<JobListSortKey>('date')
    const [sortDir, setSortDir] = React.useState<JobListSortDir>('desc')
    const [showFeedback, setShowFeedback] = React.useState<boolean>(false)
    const { used: feedbackPromptSeen, ready: feedbackPromptReady, mark: markFeedbackPromptSeen } =
        useOnce(FEEDBACK_PROMPT_SEEN_KEY)
    const { used: playerOpenedOnce, ready: playerOpenedOnceReady } = useOnce(PLAYER_OPENED_ONCE_KEY)
    const [ingressUploading, setIngressUploading] = React.useState<number>(0)

    // Ingress folder handle stored at the page level for the whole session
    const [dirHandle, setDirHandle] = React.useState<FileSystemDirectoryHandle | null>(null)
    const [dirPermission, setDirPermission] = React.useState<'granted' | 'denied' | 'prompt' | null>(null)

    const reloadStoredDirectoryHandle = React.useCallback(async () => {
        const stored = await loadDirectoryHandle()
        if (!stored) {
            setDirPermission(null)
            setDirHandle(null)
            return
        }
        try {
            const p = await stored.queryPermission({ mode: 'read' })
            setDirPermission(p || null)
            if (p === 'granted') setDirHandle(stored)
            else setDirHandle(null)
        } catch {
            setDirPermission(null)
            setDirHandle(null)
        }
    }, [])

    React.useEffect(() => {
        // Try to restore directory handle from IndexedDB on mount
        void reloadStoredDirectoryHandle()

        // Keep directory handle in sync across tabs.
        return subscribeIngressDirectoryChanged(() => reloadStoredDirectoryHandle())
    }, [reloadStoredDirectoryHandle])


    const pickDirectory = async () => {
        try {
            const handle = await window.showDirectoryPicker({ id: 'windsurf-ingress' })
            if (!handle) return
            const perm = await handle.requestPermission({ mode: 'read' })
            setDirPermission(perm || null)
            setDirHandle(handle)
            await saveDirectoryHandle(handle)
            notifyIngressDirectoryChanged()
            trackEvent('ingress_folder_selected', { permission: perm || null })
        } catch (e) {
            // user cancelled or unsupported
            const name = (e as any)?.name
            trackEvent('ingress_folder_select_failed', { name: name ? String(name) : 'unknown' })
        }
    }

    const [openingId, setOpeningId] = React.useState<string | null>(null)
    const onOpen = async (id: string) => {
        setOpeningId(id)
        try {
            trackEvent('job_open', { job_id: id })
            const detail = await refreshJobDetail(id)
            assert(!!detail.local_relative_path, 'Job must have a local relative path')
            setSelectedJob(detail)
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

    const succeededJobs = React.useMemo(() => jobs.filter(j => j.status === 'succeeded'), [jobs])

    React.useEffect(() => {
        if (!jobsInitialSyncComplete) return
        if (!feedbackPromptReady || !playerOpenedOnceReady) return
        if (feedbackPromptSeen) return
        if (!playerOpenedOnce) return
        if (succeededJobs.length <= 3) return
        setShowFeedback(true)
    }, [
        feedbackPromptReady,
        feedbackPromptSeen,
        jobsInitialSyncComplete,
        playerOpenedOnce,
        playerOpenedOnceReady,
        succeededJobs.length,
    ])

    const { showTutorial, openTutorial, tutorialModalProps } = useTutorialController({
        dirHandle,
        selectedJob,
        onPickIngressFolder: () => void pickDirectory(),
    })

    const handleFeedbackClose = React.useCallback(() => {
        setShowFeedback(false)
        markFeedbackPromptSeen()
    }, [markFeedbackPromptSeen])

    const userLabel = user?.displayName ?? (user?.email ? user.email.split('@')[0] : '')
    const headerText = userLabel
        ? t('screens.analyzer.header.taglineWithName', { name: userLabel })
        : t('screens.analyzer.header.tagline')
    const shouldShowFeedback = showFeedback && !selectedJob
    const containerVars = {
        '--analyzer-side-offset': 'max(1rem, calc((100vw - 1400px) / 2 + 1rem))',
        '--analyzer-bottom-offset': 'calc(1rem + var(--analytics-consent-offset, 0px))',
    } as React.CSSProperties

    const confirmLeaveIfUploading = React.useCallback(() => {
        if (ingressUploading <= 0) return true
        const warning = t('components.ingressWidget.leaveWarning')
        return window.confirm(warning)
    }, [ingressUploading, t])

    const handleGoHome = React.useCallback(() => {
        if (!confirmLeaveIfUploading()) return
        onGoHome()
    }, [confirmLeaveIfUploading, onGoHome])

    const handleGoPricing = React.useCallback(() => {
        if (!confirmLeaveIfUploading()) return
        onGoPricing()
    }, [confirmLeaveIfUploading, onGoPricing])

    return (
        <div className="min-h-dvh bg-white text-slate-900" style={containerVars}>
            <UnsupportedBrowserBanner />
            <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                    <LogoButton onClick={handleGoHome} />

                    <div className="text-sm text-slate-600">{headerText}</div>

                    <div className="flex-1" />

                    <div className="flex items-center gap-2 flex-wrap">
                        <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => {
                                openTutorial('header', null, 'what')
                            }}
                            text={t('screens.analyzer.actions.tutorial')}
                        />
                        <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => setShowHelp(true)}
                            text={t('screens.analyzer.actions.help')}
                        />
                        <Button
                            size="sm"
                            variant="primary"
                            onClick={() => {
                                trackEvent('open_feedback')
                                setShowFeedback(true)
                            }}
                            text={t('screens.analyzer.actions.feedback')}
                        />
                        <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => {
                                trackEvent('open_settings')
                                setShowSettings(true)
                            }}
                            aria-label={t('screens.analyzer.actions.settings')}
                            title={t('screens.analyzer.actions.settings')}
                        >
                            <img src={settingsIcon} alt="Settings" className="h-4 w-4 fill-white" />
                        </Button>
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
                                    onClick={() => openTutorial('empty_state', null, 'what')}
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
                    onDeleteJobs={deleteJobs}
                    openingId={openingId}
                    dirHandle={dirHandle}
                    initialSyncComplete={jobsInitialSyncComplete}
                />

                {selectedJob && (
                    <PlayerModal
                        job={selectedJob}
                        videoSource={{ kind: 'ingress', dirHandle }}
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
                        onReport={reportJob}
                    />
                )}
                {showSettings && <SettingsModal onClose={() => setShowSettings(false)} onLogout={logout} />}
                {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
                {shouldShowFeedback && <FeedbackModal onClose={handleFeedbackClose} />}
                {showTutorial && <AnalyzerTutorialModal {...tutorialModalProps} />}

                <IngressWidget
                    dirHandle={dirHandle}
                    dirPermission={dirPermission}
                    onPickDirectory={pickDirectory}
                    authorizedFetch={authorizedFetch}
                    jobs={jobs}
                    enabled={jobsInitialSyncComplete}
                    onUploadingChange={setIngressUploading}
                />

                {/* Beta badge: bottom-left, subtle (brand) */}
                <div
                    className="fixed"
                    style={{
                        left: 'var(--analyzer-side-offset, 1rem)',
                        bottom: 'var(--analyzer-bottom-offset, 1rem)',
                    }}
                >
                    <Button
                        type="button"
                        variant="unstyled"
                        size="none"
                        onClick={() => {
                            trackEvent('open_pricing_from_beta_badge')
                            handleGoPricing()
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
                    </Button>
                </div>
            </main>
        </div>
    )
}
