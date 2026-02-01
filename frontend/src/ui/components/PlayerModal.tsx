/**
 * @module PlayerModal
 * This module provides the primary modal interface for the video player,
 * integrating job metadata, player controls, and feedback reporting.
 */

import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { Modal } from './Modal'
import { Button } from './Button'
import { KeyboardShortcutsModal } from './KeyboardShortcutsModal'
import { Player } from '../player/Player'
import type { PlayerState } from '../player/state'
import { JobDetail, ReportType } from '../types'
import { Spinner } from './Spinner'
import { loadSetting, saveSetting } from '../utils/idb'
import { VideoSource } from '../player/videoSource'
import { getFileByRelativePath } from '../utils/fsAccess'

const PLAYER_DISABLE_OVERVIEW_STABILIZATION_KEY = 'player.disableOverviewStabilization.v1'

/**
 * A modal component that hosts the video player for reviewing job results.
 *
 * It manages the player lifecycle, metadata loading (such as video duration),
 * and provides UI for drawing, stabilization toggles, and reporting issues.
 */
export const PlayerModal: React.FC<{
    /** The job details containing tracking data and metadata to be displayed. */
    job: JobDetail
    /** The source of the video file, either a direct File object or a directory handle. */
    videoSource: VideoSource
    /** Callback invoked when the modal is requested to close. */
    onClose: () => void
    /** Callback invoked when a user submits a feedback report for the current job. */
    onReport: (id: string, type: ReportType, message: string) => void
    /** Optional callback to navigate to the next job in the current sequence. */
    onOpenNextJob?: () => void
    /** Optional callback to navigate to the previous job in the current sequence. */
    onOpenPrevJob?: () => void
    /** Whether to display tutorial tips for the demo experience. Defaults to false. */
    showDemoTutorialTips?: boolean
}> = ({ job, videoSource, onClose, onReport, onOpenNextJob, onOpenPrevJob, showDemoTutorialTips = false }) => {
    const { t } = useTranslation()
    const [showShortcuts, setShowShortcuts] = React.useState<boolean>(false)
    const [showReport, setShowReport] = React.useState<boolean>(false)
    const [showReportThanks, setShowReportThanks] = React.useState<boolean>(false)
    const [showInfo, setShowInfo] = React.useState<boolean>(false)
    const [drawMode, setDrawMode] = React.useState<boolean>(false)
    const [player, setPlayer] = React.useState<PlayerState | null>(null)
    const [disableOverviewStabilization, setDisableOverviewStabilization] = React.useState<boolean>(false)
    const [showDemoTips, setShowDemoTips] = React.useState<boolean>(showDemoTutorialTips)
    const [durationSeconds, setDurationSeconds] = React.useState<number | null>(null)

    React.useEffect(() => {
        loadSetting<boolean>(PLAYER_DISABLE_OVERVIEW_STABILIZATION_KEY).then(saved => {
            setDisableOverviewStabilization(!!saved)
        })
    }, [])

    React.useEffect(() => {
        setDrawMode(false)
        setPlayer(null)
    }, [job.id])

    const toggleDrawMode = React.useCallback(() => {
        setDrawMode(mode => !mode)
    }, [])

    const toggleOverviewStabilization = React.useCallback(() => {
        setDisableOverviewStabilization(prev => {
            const next = !prev
            void saveSetting(PLAYER_DISABLE_OVERVIEW_STABILIZATION_KEY, next)
            return next
        })
    }, [])

    const durationKey = React.useMemo(() => {
        if (videoSource.kind === 'file') {
            const file = videoSource.file
            return `${file.name}|${file.type}|${file.size}|${file.lastModified}`
        }
        return `${job.id}|${job.local_relative_path ?? ''}`
    }, [job.id, job.local_relative_path, videoSource.kind === 'file' ? videoSource.file : null, videoSource.kind])

    const ingressDirHandle = videoSource.kind === 'ingress' ? videoSource.dirHandle : null

    React.useEffect(() => {
        let revoked: string | null = null
        let cancelled = false
        const loadDuration = async () => {
            setDurationSeconds(null)
            try {
                let file: File | null = null
                if (videoSource.kind === 'file') {
                    file = videoSource.file
                } else {
                    const dh = videoSource.dirHandle
                    if (!dh) return
                    const rel = job.local_relative_path
                    if (!rel) return
                    file = await getFileByRelativePath(dh, rel)
                }
                if (!file || cancelled) return

                const url = URL.createObjectURL(file)
                revoked = url
                const video = document.createElement('video')
                video.preload = 'metadata'
                video.src = url
                await new Promise<void>((resolve, reject) => {
                    const onLoaded = () => resolve()
                    const onError = () => reject(new Error('failed_to_load_video_metadata'))
                    video.addEventListener('loadedmetadata', onLoaded, { once: true })
                    video.addEventListener('error', onError, { once: true })
                })
                if (cancelled) return
                const d = Number.isFinite(video.duration) ? video.duration : NaN
                if (Number.isFinite(d) && d > 0) setDurationSeconds(d)
            } catch {
                // ignore metadata failures
            } finally {
                if (revoked) URL.revokeObjectURL(revoked)
            }
        }
        void loadDuration()
        return () => {
            cancelled = true
            if (revoked) URL.revokeObjectURL(revoked)
        }
    }, [durationKey, ingressDirHandle, videoSource.kind])

    const title =
        videoSource.kind === 'file'
            ? videoSource.file.name.replace(/\.(mp4|hevc|mov|mkv)$/i, '')
            : job.local_relative_path?.replace(/\.(mp4|hevc|mov|mkv)$/i, '') ?? job.id ?? t('common.notAvailable')

    const [moreOpen, setMoreOpen] = React.useState(false)
    const moreRef = React.useRef<HTMLDivElement | null>(null)

    const riderCount = job.tracks?.length ?? 0
    const createdAtLabel = React.useMemo(() => {
        const raw = job.created_at
        if (!raw) return null
        try {
            const d = new Date(raw)
            if (!Number.isFinite(d.getTime())) return null
            return d.toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
        } catch {
            return null
        }
    }, [job.created_at])

    React.useEffect(() => {
        const onPointerDown = (event: MouseEvent) => {
            const el = moreRef.current
            if (!el) return
            if (el.contains(event.target as Node)) return
            setMoreOpen(false)
        }
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') setMoreOpen(false)
        }
        document.addEventListener('mousedown', onPointerDown)
        document.addEventListener('keydown', onKeyDown)
        return () => {
            document.removeEventListener('mousedown', onPointerDown)
            document.removeEventListener('keydown', onKeyDown)
        }
    }, [])

    return (
        <>
            <Modal
                key={job.id}
                onClose={onClose}
                closeOnEscape={player?.mode !== 'detailed'}
                title={title}
                showCloseButton={false}
                additionalHeader={
                    <div className="flex items-center gap-2">
                        <Button
                            onClick={toggleOverviewStabilization}
                            title={t('components.playerModal.actions.overviewStabilization.title')}
                            text={
                                disableOverviewStabilization
                                    ? t('components.playerModal.actions.overviewStabilization.labelOff')
                                    : t('components.playerModal.actions.overviewStabilization.labelOn')
                            }
                            variant={disableOverviewStabilization ? 'ghost' : 'brandOutline'}
                        />
                        <Button
                            onClick={toggleDrawMode}
                            title={t('components.playerModal.actions.draw.title')}
                            text={t('components.playerModal.actions.draw.label')}
                            variant={drawMode ? 'brandOutline' : 'ghost'}
                        />
                        <div ref={moreRef} className="relative">
                            <Button
                                type="button"
                                variant="ghost"
                                aria-haspopup="menu"
                                aria-expanded={moreOpen}
                                onClick={() => setMoreOpen(open => !open)}
                                title={t('common.more')}
                                className="px-2"
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
                                    <path d="M5 10.5a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3Zm7 0a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3Zm7 0a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3Z" />
                                </svg>
                            </Button>
                            {moreOpen && (
                                <div
                                    role="menu"
                                    aria-label={t('common.more')}
                                    className="absolute right-0 mt-2 w-40 rounded-md border border-slate-200 bg-white shadow-lg z-50 overflow-hidden"
                                >
                                    <Button
                                        type="button"
                                        variant="unstyled"
                                        size="none"
                                        role="menuitem"
                                        onClick={() => {
                                            setMoreOpen(false)
                                            setShowInfo(v => !v)
                                        }}
                                        className="w-full px-3 py-2 text-xs text-left text-slate-700 hover:bg-slate-50"
                                    >
                                        {t('common.info')}
                                    </Button>
                                    <Button
                                        type="button"
                                        variant="unstyled"
                                        size="none"
                                        role="menuitem"
                                        onClick={() => {
                                            setMoreOpen(false)
                                            setShowDemoTips(true)
                                        }}
                                        className="w-full px-3 py-2 text-xs text-left text-slate-700 hover:bg-slate-50"
                                    >
                                        {t('components.playerModal.actions.help.label')}
                                    </Button>
                                    <Button
                                        type="button"
                                        variant="unstyled"
                                        size="none"
                                        role="menuitem"
                                        onClick={() => {
                                            setMoreOpen(false)
                                            setShowShortcuts(true)
                                        }}
                                        className="w-full px-3 py-2 text-xs text-left text-slate-700 hover:bg-slate-50"
                                    >
                                        {t('components.playerModal.actions.shortcuts.label')}
                                    </Button>
                                    <Button
                                        type="button"
                                        variant="unstyled"
                                        size="none"
                                        role="menuitem"
                                        onClick={() => {
                                            setMoreOpen(false)
                                            setShowReport(true)
                                        }}
                                        className="w-full px-3 py-2 text-xs text-left text-slate-700 hover:bg-slate-50"
                                    >
                                        {t('components.playerModal.actions.report.label')}
                                    </Button>
                                </div>
                            )}
                        </div>
                        <Button
                            onClick={onClose}
                            variant="ghost"
                            title={t('common.close')}
                            aria-label={t('common.close')}
                            className="px-2"
                        >
                            <img src="/icons/close.svg" alt="" className="h-4 w-4" />
                        </Button>
                    </div>
                }
            >
                <div className="relative w-[96vw] h-[92vh] bg-white text-black rounded-md shadow-xl overflow-hidden">
                    {showInfo && (
                        <div className="absolute z-30 top-3 left-3 w-[min(320px,92vw)] rounded-xl border border-slate-200 bg-white/95 backdrop-blur shadow-sm p-3">
                            <div className="flex items-start gap-2">
                                <div className="flex-1 min-w-0">
                                    <div className="text-xs font-semibold text-slate-900">{t('common.info')}</div>
                                    <div className="mt-1 text-xs text-slate-700 space-y-1">
                                        <div className="flex items-center justify-between gap-3">
                                            <span className="text-slate-500">Riders</span>
                                            <span className="tabular-nums text-slate-900">{riderCount}</span>
                                        </div>
                                        {createdAtLabel && (
                                            <div className="flex items-center justify-between gap-3">
                                                <span className="text-slate-500">Processed</span>
                                                <span className="text-slate-900">{createdAtLabel}</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                                <button
                                    type="button"
                                    className="shrink-0 rounded-md p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
                                    onClick={() => setShowInfo(false)}
                                    aria-label={t('common.close')}
                                >
                                    <img src="/icons/close.svg" alt="" className="h-4 w-4" />
                                </button>
                            </div>
                        </div>
                    )}
                    {showDemoTips && (
                        <div className="absolute z-30 top-3 right-3 w-[min(280px,92vw)] rounded-xl border border-slate-200 bg-white/95 backdrop-blur shadow-sm p-3">
                            <div className="flex items-start gap-2">
                                <div className="flex-1 min-w-0">
                                    <div className="text-xs font-semibold text-slate-900">
                                        {t('components.analyzerTutorialModal.steps.reviewRiding.title')}
                                    </div>
                                    <div className="mt-1 text-xs text-slate-700 space-y-2">
                                        <div>
                                            <Trans
                                                i18nKey="components.analyzerTutorialModal.steps.reviewRiding.overviewIntro"
                                                components={{ b: <span className="font-semibold text-slate-900" /> }}
                                            />
                                            <ul className="mt-1 list-disc pl-5 space-y-0.5">
                                                <li>{t('components.analyzerTutorialModal.steps.reviewRiding.overviewBullets.stabilized')}</li>
                                                <li>{t('components.analyzerTutorialModal.steps.reviewRiding.overviewBullets.allRiders')}</li>
                                            </ul>
                                        </div>
                                        <div>
                                            <Trans
                                                i18nKey="components.analyzerTutorialModal.steps.reviewRiding.focusedIntro"
                                                components={{ b: <span className="font-semibold text-slate-900" /> }}
                                            />
                                            <ul className="mt-1 list-disc pl-5 space-y-0.5">
                                                <li>{t('components.analyzerTutorialModal.steps.reviewRiding.focusedBullets.centered')}</li>
                                                <li>{t('components.analyzerTutorialModal.steps.reviewRiding.focusedBullets.cameraMotionRemoved')}</li>
                                            </ul>
                                        </div>
                                        <div className="text-[11px] text-slate-500">
                                            <Trans
                                                i18nKey="components.analyzerTutorialModal.steps.reviewRiding.tip2"
                                                components={{ b: <span className="font-semibold text-slate-700" /> }}
                                            />
                                        </div>
                                    </div>

                                    <div className="mt-3 border-t border-slate-200 pt-2">
                                        <div className="text-xs font-semibold text-slate-900">
                                            {t('components.playerModal.actions.report.label')}
                                        </div>
                                        <ul className="mt-1 list-disc pl-5 space-y-0.5 text-xs text-slate-700">
                                            <li>
                                                <Trans
                                                    i18nKey="components.analyzerTutorialModal.steps.feedbackReports.bullets.report"
                                                    components={{ b: <span className="font-semibold text-slate-900" /> }}
                                                />
                                            </li>
                                        </ul>
                                        <div className="mt-1 text-[11px] text-slate-500">
                                            {t('components.analyzerTutorialModal.steps.feedbackReports.muted')}
                                        </div>
                                    </div>
                                </div>

                                <button
                                    type="button"
                                    className="shrink-0 rounded-md p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
                                    onClick={() => setShowDemoTips(false)}
                                    aria-label={t('common.close')}
                                >
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M18.3 5.71a1 1 0 0 0-1.41 0L12 10.59 7.11 5.7A1 1 0 0 0 5.7 7.11L10.59 12l-4.9 4.89a1 1 0 1 0 1.41 1.42L12 13.41l4.89 4.9a1 1 0 0 0 1.42-1.41L13.41 12l4.9-4.89a1 1 0 0 0-.01-1.4Z" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    )}
                    <div className="w-full h-full overflow-hidden">
                        <Player
                            key={job.id}
                            job={job}
                            videoSource={videoSource}
                            onClose={onClose}
                            onReport={onReport}
                            onOpenNextJob={onOpenNextJob}
                            onOpenPrevJob={onOpenPrevJob}
                            drawMode={drawMode}
                            onToggleDrawMode={toggleDrawMode}
                            disableOverviewStabilization={disableOverviewStabilization}
                            player={player}
                            setPlayer={setPlayer}
                            durationSeconds={durationSeconds}
                        />
                    </div>
                </div>
            </Modal>

            {showShortcuts && <KeyboardShortcutsModal onClose={() => setShowShortcuts(false)} />}
            {showReport && (
                <ReportVideoModal
                    job={job}
                    onClose={() => setShowReport(false)}
                    onReport={async (type, message) => {
                        await onReport(job.id, type, message)
                    }}
                    onSubmitted={() => {
                        setShowReport(false)
                        setShowReportThanks(true)
                    }}
                />
            )}
            {showReportThanks && (
                <Modal
                    onClose={() => setShowReportThanks(false)}
                    title={t('components.playerModal.report.thanksTitle')}
                >
                    <div className="p-4 text-sm text-slate-600">{t('components.playerModal.report.thanksBody')}</div>
                    <div className="px-4 pb-4 flex justify-end">
                        <Button text={t('common.done')} onClick={() => setShowReportThanks(false)} />
                    </div>
                </Modal>
            )}
        </>
    )
}

const ReportVideoModal: React.FC<{
    job: JobDetail
    onClose: () => void
    onReport: (type: ReportType, message: string) => Promise<void> | void
    onSubmitted: () => void
}> = ({ job, onClose, onReport, onSubmitted }) => {
    const { t } = useTranslation()
    const [type, setType] = React.useState<ReportType>('missed_detection')
    const [message, setMessage] = React.useState<string>('')
    const [isSubmitting, setIsSubmitting] = React.useState<boolean>(false)
    const [error, setError] = React.useState<string | null>(null)

    const canSubmit = !isSubmitting && message.trim().length > 0

    if (isSubmitting) {
        return (
            <Modal onClose={onClose} title={t('components.playerModal.report.processingTitle')}>
                <div className="p-6 flex flex-col items-center gap-3">
                    <Spinner size="medium" />
                    <div className="text-sm text-slate-600">{t('components.playerModal.report.processingBody')}</div>
                </div>
            </Modal>
        )
    }

    return (
        <Modal onClose={onClose} title={t('components.playerModal.report.title')}>
            <div className="p-4 space-y-4 max-w-[720px]">
                <div className="text-sm text-slate-600">
                    <Trans
                        i18nKey="components.playerModal.report.subtitle"
                        components={{ strong: <span className="font-medium text-slate-800" /> }}
                        values={{ target: job.local_relative_path ?? job.id }}
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-slate-900 mb-1">
                        {t('components.playerModal.report.issueType')}
                    </label>
                    <select
                        className="w-full bg-white border border-slate-200 rounded-md p-2 text-slate-900"
                        value={type}
                        onChange={e => setType(e.target.value as ReportType)}
                        disabled={isSubmitting}
                    >
                        <option value="missed_detection">
                            {t('components.playerModal.report.issueOptions.missedDetection')}
                        </option>
                        <option value="false_association">
                            {t('components.playerModal.report.issueOptions.falseAssociation')}
                        </option>
                        <option value="visual_problem">
                            {t('components.playerModal.report.issueOptions.visualProblems')}
                        </option>
                        <option value="other">{t('components.playerModal.report.issueOptions.other')}</option>
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-slate-900 mb-1">
                        {t('components.playerModal.report.descriptionLabel')}
                    </label>
                    <textarea
                        className="w-full min-h-28 bg-white border border-slate-200 rounded-md p-2 text-slate-900"
                        placeholder={t('components.playerModal.report.placeholder')}
                        value={message}
                        onChange={e => setMessage(e.target.value)}
                        disabled={isSubmitting}
                    />
                    <div className="mt-2 text-xs text-slate-500">{t('components.playerModal.report.tip')}</div>
                </div>

                {error && <div className="text-sm text-red-700">{error}</div>}

                <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-200">
                    <Button variant="ghost" onClick={onClose} text={t('common.cancel')} disabled={isSubmitting} />
                    <Button
                        variant="primary"
                        text={
                            isSubmitting
                                ? t('components.playerModal.report.sending')
                                : t('components.playerModal.report.send')
                        }
                        disabled={!canSubmit}
                        onClick={async () => {
                            setError(null)
                            setIsSubmitting(true)
                            try {
                                await onReport(type, message.trim())
                                setMessage('')
                                onSubmitted()
                            } catch (e: any) {
                                setError(e?.message || String(e))
                            } finally {
                                setIsSubmitting(false)
                            }
                        }}
                    />
                </div>
            </div>
        </Modal>
    )
}
