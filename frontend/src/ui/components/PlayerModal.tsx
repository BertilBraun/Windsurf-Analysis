import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { Modal } from './Modal'
import { Button } from './Button'
import { KeyboardShortcutsModal } from './KeyboardShortcutsModal'
import { CanvasPlayer } from '../player/CanvasPlayer'
import { JobDetail, ReportType } from '../types'

export const PlayerModal: React.FC<{
    job: JobDetail
    dirHandle: FileSystemDirectoryHandle | null
    onClose: () => void
    onDelete: (id: string) => void
    onReport: (id: string, type: ReportType, message: string) => void
    onOpenNextJob?: () => void
    onOpenPrevJob?: () => void
    deletingId?: string | null
}> = ({ job, dirHandle, onClose, onDelete, onReport, onOpenNextJob, onOpenPrevJob, deletingId }) => {
    const { t } = useTranslation()
    const [showShortcuts, setShowShortcuts] = React.useState<boolean>(false)
    const [showReport, setShowReport] = React.useState<boolean>(false)
    const [drawMode, setDrawMode] = React.useState<boolean>(false)

    React.useEffect(() => {
        setDrawMode(false)
    }, [job.id])

    const toggleDrawMode = React.useCallback(() => {
        setDrawMode(mode => !mode)
    }, [])

    return (
        <>
            <Modal
                key={job.id}
                onClose={onClose}
                title={job.local_relative_path?.replace(/\.mp4$/i, '') ?? t('common.notAvailable')}
                additionalHeader={
                    <>
                        <Button
                            onClick={toggleDrawMode}
                            title={t('components.playerModal.actions.draw.title')}
                            text={t('components.playerModal.actions.draw.label')}
                            variant={drawMode ? 'primary' : 'secondary'}
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
                        <Button
                            onClick={() => onDelete(job.id)}
                            title={t('components.playerModal.actions.delete.title')}
                            text={t('components.playerModal.actions.delete.label')}
                            isPending={deletingId === job.id}
                        />
                    </>
                }
            >
                <div className="relative w-[96vw] h-[92vh] bg-white text-black rounded-md shadow-xl overflow-hidden">
                    <div className="w-full h-full overflow-hidden">
                        <CanvasPlayer
                            key={job.id}
                            job={job}
                            dirHandle={dirHandle}
                            onClose={onClose}
                            onDelete={onDelete}
                            onReport={onReport}
                            onOpenNextJob={onOpenNextJob}
                            onOpenPrevJob={onOpenPrevJob}
                            drawMode={drawMode}
                            onToggleDrawMode={toggleDrawMode}
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
                />
            )}
        </>
    )
}

const ReportVideoModal: React.FC<{
    job: JobDetail
    onClose: () => void
    onReport: (type: ReportType, message: string) => Promise<void> | void
}> = ({ job, onClose, onReport }) => {
    const { t } = useTranslation()
    const [type, setType] = React.useState<ReportType>('missed_detection')
    const [message, setMessage] = React.useState<string>('')
    const [isSubmitting, setIsSubmitting] = React.useState<boolean>(false)
    const [error, setError] = React.useState<string | null>(null)

    const canSubmit = !isSubmitting && message.trim().length > 0

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
                                onClose()
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
