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

const PLAYER_DISABLE_OVERVIEW_STABILIZATION_KEY = 'player.disableOverviewStabilization.v1'

export const PlayerModal: React.FC<{
    job: JobDetail
    videoSource: VideoSource
    onClose: () => void
    onReport: (id: string, type: ReportType, message: string) => void
    onOpenNextJob?: () => void
    onOpenPrevJob?: () => void
}> = ({ job, videoSource, onClose, onReport, onOpenNextJob, onOpenPrevJob }) => {
    const { t } = useTranslation()
    const [showShortcuts, setShowShortcuts] = React.useState<boolean>(false)
    const [showReport, setShowReport] = React.useState<boolean>(false)
    const [showReportThanks, setShowReportThanks] = React.useState<boolean>(false)
    const [drawMode, setDrawMode] = React.useState<boolean>(false)
    const [player, setPlayer] = React.useState<PlayerState | null>(null)
    const [disableOverviewStabilization, setDisableOverviewStabilization] = React.useState<boolean>(false)

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

    const title =
        videoSource.kind === 'file'
            ? videoSource.file.name.replace(/\.mp4$/i, '')
            : job.local_relative_path?.replace(/\.mp4$/i, '') ?? job.id ?? t('common.notAvailable')

    return (
        <>
            <Modal
                key={job.id}
                onClose={onClose}
                closeOnEscape={player?.mode !== 'detailed'}
                title={title}
                additionalHeader={
                    <>
                        <Button
                            onClick={toggleDrawMode}
                            title={t('components.playerModal.actions.draw.title')}
                            text={t('components.playerModal.actions.draw.label')}
                            variant={drawMode ? 'primary' : 'secondary'}
                        />
                        <Button
                            onClick={toggleOverviewStabilization}
                            title={t('components.playerModal.actions.overviewStabilization.title')}
                            text={
                                disableOverviewStabilization
                                    ? t('components.playerModal.actions.overviewStabilization.labelOff')
                                    : t('components.playerModal.actions.overviewStabilization.labelOn')
                            }
                            variant={disableOverviewStabilization ? 'secondary' : 'primary'}
                        />
                        <Button
                            onClick={() => setShowShortcuts(true)}
                            title={t('components.playerModal.actions.shortcuts.title')}
                            text={t('components.playerModal.actions.shortcuts.label')}
                        />
                        <Button
                            onClick={() => setShowReport(true)}
                            title={t('components.playerModal.actions.report.title')}
                            text={t('components.playerModal.actions.report.label')}
                        />
                    </>
                }
            >
                <div className="relative w-[96vw] h-[92vh] bg-white text-black rounded-md shadow-xl overflow-hidden">
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
