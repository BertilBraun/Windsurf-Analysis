import React from 'react'
import { useTranslation } from 'react-i18next'
import { ReportType } from '../types'
import { Button } from './Button'
import { Modal } from './Modal'

type Props = {
    onClose: () => void
    onSubmit: (jobId: string, type: ReportType, message: string) => Promise<void>
    jobId: string | null
}

export const FeedbackModal: React.FC<Props> = ({ onClose, onSubmit, jobId }) => {
    const { t } = useTranslation()
    const [message, setMessage] = React.useState<string>('')
    const [isSubmitting, setIsSubmitting] = React.useState<boolean>(false)
    const [error, setError] = React.useState<string | null>(null)

    const canSubmit = !isSubmitting && !!jobId && message.trim().length > 0

    return (
        <Modal onClose={onClose} title={t('screens.analyzer.feedback.title')}>
            <div className="p-4 space-y-3">
                <div className="text-sm text-slate-600">{t('screens.analyzer.feedback.body')}</div>
                <textarea
                    className="w-full min-h-28 bg-white border border-slate-200 rounded-md p-2 text-slate-900"
                    placeholder={t('screens.analyzer.feedback.placeholder')}
                    value={message}
                    onChange={e => setMessage(e.target.value)}
                    disabled={isSubmitting}
                />
                {error && <div className="text-sm text-red-700">{error}</div>}
                <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-200">
                    <Button variant="ghost" onClick={onClose} text={t('common.cancel')} disabled={isSubmitting} />
                    <Button
                        variant="primary"
                        text={t('screens.analyzer.feedback.send')}
                        disabled={isSubmitting || !canSubmit}
                        isPending={isSubmitting}
                        onClick={async () => {
                            if (!jobId) return
                            setError(null)
                            setIsSubmitting(true)
                            try {
                                await onSubmit(jobId, 'feedback', message.trim())
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
