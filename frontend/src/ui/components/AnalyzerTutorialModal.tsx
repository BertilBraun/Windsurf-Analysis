import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { Modal } from './Modal'
import { Button } from './Button'

type Step = {
    key: string
    title: string
    body: React.ReactNode
}

export type AnalyzerTutorialModalProps = {
    onClose: () => void
    stepIndex: number
    onStepIndexChange: (next: number) => void
    onPickIngressFolder: () => void
    ingressFolderName: string | null
    stepKeys?: string[] | null
}

export const AnalyzerTutorialModal: React.FC<AnalyzerTutorialModalProps> = ({
    onClose,
    stepIndex,
    onStepIndexChange,
    onPickIngressFolder,
    ingressFolderName,
    stepKeys,
}) => {
    const { t } = useTranslation()
    const steps: Step[] = React.useMemo(
        () => [
            {
                key: 'what',
                title: t('components.analyzerTutorialModal.steps.what.title'),
                body: (
                    <div className="space-y-3">
                        <p className="text-sm text-slate-700 leading-6">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.what.body"
                                components={{ b: <b /> }}
                            />
                        </p>
                        <div className="text-xs text-slate-500">
                            {t('components.analyzerTutorialModal.steps.what.bodyMuted')}
                        </div>
                        <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-slate-700">
                            <div className="text-xs text-slate-600">
                                {t('components.analyzerTutorialModal.steps.what.footageNote1')}
                            </div>
                            <div className="text-xs text-slate-600">
                                {t('components.analyzerTutorialModal.steps.what.footageNote2')}
                            </div>
                        </div>
                    </div>
                ),
            },
            {
                key: 'watch-folder',
                title: t('components.analyzerTutorialModal.steps.watchFolder.title'),
                body: (
                    <div className="space-y-3">
                        <p className="text-sm text-slate-700 leading-6">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.watchFolder.body"
                                components={{ b: <b /> }}
                            />
                        </p>
                        <div className="rounded-xl border border-slate-200 bg-white p-3 text-xs text-slate-600">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.watchFolder.tip"
                                components={{ b: <b /> }}
                            />
                        </div>
                        <div className="text-sm text-slate-700 leading-6">
                            {t('components.analyzerTutorialModal.steps.watchFolder.permissions')}
                        </div>
                        <div className="text-sm text-slate-700 leading-6">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.watchFolder.safety"
                                components={{ b: <b /> }}
                            />
                        </div>

                        <div className="pt-2 flex flex-col items-center gap-2">
                            <Button
                                variant={ingressFolderName ? 'secondary' : 'primary'}
                                onClick={() => onPickIngressFolder()}
                                text={
                                    ingressFolderName
                                        ? t('common.actions.changeFolder')
                                        : t('components.analyzerTutorialModal.steps.watchFolder.button')
                                }
                            />
                            {ingressFolderName ? (
                                <div className="text-xs text-slate-600 text-center">
                                    <div className="font-semibold text-slate-900">
                                        {t('components.analyzerTutorialModal.steps.watchFolder.currentFolder', {
                                            name: ingressFolderName,
                                        })}
                                    </div>
                                    <div className="text-slate-600">
                                        {t('components.analyzerTutorialModal.steps.watchFolder.status')}
                                    </div>
                                </div>
                            ) : null}
                        </div>
                    </div>
                ),
            },
            {
                key: 'add-videos',
                title: t('components.analyzerTutorialModal.steps.addVideos.title'),
                body: (
                    <div className="space-y-3">
                        <p className="text-sm text-slate-700 leading-6">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.addVideos.body"
                                components={{ b: <b /> }}
                            />
                        </p>
                        <p className="text-sm text-slate-700 leading-6">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.addVideos.subfolders"
                                components={{ b: <b /> }}
                            />
                        </p>
                        <div className="text-xs text-slate-500">
                            {t('components.analyzerTutorialModal.steps.addVideos.muted')}
                        </div>
                    </div>
                ),
            },
            {
                key: 'review-riding',
                title: t('components.analyzerTutorialModal.steps.reviewRiding.title'),
                body: (
                    <div className="space-y-3">
                        <p className="text-sm text-slate-700 leading-6">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.reviewRiding.overview"
                                components={{ b: <b /> }}
                            />
                        </p>
                        <p className="text-sm text-slate-700 leading-6">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.reviewRiding.focused"
                                components={{ b: <b /> }}
                            />
                        </p>
                        <div className="text-xs text-slate-500">
                            {t('components.analyzerTutorialModal.steps.reviewRiding.tip')}
                        </div>
                    </div>
                ),
            },
            {
                key: 'review-tools',
                title: t('components.analyzerTutorialModal.steps.reviewTools.title'),
                body: (
                    <div className="space-y-3">
                        <ul className="list-disc pl-5 space-y-1 text-sm text-slate-700">
                            <li>{t('components.analyzerTutorialModal.steps.reviewTools.bullets.draw')}</li>
                            <li>{t('components.analyzerTutorialModal.steps.reviewTools.bullets.slow')}</li>
                            <li>{t('components.analyzerTutorialModal.steps.reviewTools.bullets.export')}</li>
                        </ul>
                    </div>
                ),
            },
            {
                key: 'feedback-reports',
                title: t('components.analyzerTutorialModal.steps.feedbackReports.title'),
                body: (
                    <div className="space-y-3">
                        <p className="text-sm text-slate-700 leading-6">
                            {t('components.analyzerTutorialModal.steps.feedbackReports.body')}
                        </p>
                        <ul className="list-disc pl-5 space-y-1 text-sm text-slate-700">
                            <li>
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.feedbackReports.bullets.report"
                                    components={{ b: <b /> }}
                                />
                            </li>
                            <li>
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.feedbackReports.bullets.feedback"
                                    components={{ b: <b /> }}
                                />
                            </li>
                        </ul>
                        <div className="text-xs text-slate-500">
                            {t('components.analyzerTutorialModal.steps.feedbackReports.muted')}
                        </div>
                    </div>
                ),
            },
        ],
        [ingressFolderName, onPickIngressFolder, t]
    )

    const visibleSteps = React.useMemo(() => {
        if (!stepKeys || stepKeys.length === 0) return steps
        const allowed = new Set(stepKeys)
        const filtered = steps.filter(step => allowed.has(step.key))
        return filtered.length > 0 ? filtered : steps
    }, [stepKeys, steps])

    // Step state is controlled by the parent (session-only, no persistence).
    // This lets users close/reopen the modal and continue where they left off,
    // but a page refresh will reset back to step 1.
    const safeIdx = Math.min(Math.max(0, stepIndex || 0), visibleSteps.length - 1)
    const step = visibleSteps[safeIdx] ?? visibleSteps[0]!
    const isFirst = safeIdx === 0
    const isLast = safeIdx === visibleSteps.length - 1

    // Set global style var(--z-ingress-widget) so the Watch Folder widget is highlighted.
    if (step.key === 'watch-folder') {
        document.documentElement.style.setProperty('--z-ingress-widget', '1000')
    } else {
        document.documentElement.style.setProperty('--z-ingress-widget', 'auto')
    }

    return (
        <>
            <Modal
                onClose={onClose}
                title={t('components.analyzerTutorialModal.title')}
                contentClassName="rounded-2xl border border-slate-200 bg-white shadow-xl w-[760px] max-w-[96vw]"
                additionalHeader={
                    <div className="flex items-center gap-2">
                        <div className="text-xs text-slate-600">
                            {t('components.analyzerTutorialModal.stepLabel', {
                                current: safeIdx + 1,
                                total: visibleSteps.length,
                            })}
                        </div>
                        <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => onStepIndexChange(0)}
                            text={t('components.analyzerTutorialModal.restart')}
                            title={t('components.analyzerTutorialModal.restartTitle')}
                        />
                    </div>
                }
            >
                <div className="p-4 max-h-[72vh] overflow-auto">
                    <h3 className="m-0 text-lg font-semibold text-slate-900">{step.title}</h3>
                    <div className="mt-3 text-sm text-slate-700 leading-6">{step.body}</div>
                </div>

                <div className="px-4 py-3 border-t border-slate-200 flex items-center justify-between gap-2">
                    <Button
                        variant="ghost"
                        size="sm"
                        disabled={isFirst}
                        onClick={() => onStepIndexChange(Math.max(0, safeIdx - 1))}
                        text={t('common.back')}
                    />
                    <div className="flex items-center gap-2">
                        {!isLast ? (
                            <Button
                                variant="primary"
                                size="sm"
                                onClick={() => onStepIndexChange(Math.min(visibleSteps.length - 1, safeIdx + 1))}
                                text={t('common.next')}
                            />
                        ) : (
                            <Button variant="primary" size="sm" onClick={onClose} text={t('common.done')} />
                        )}
                    </div>
                </div>
            </Modal>

        </>
    )
}
