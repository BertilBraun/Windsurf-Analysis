import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { Modal } from './Modal'
import { Button } from './Button'
import { Text, TextStack, TextStrong } from './Typography'

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
    const pickRequestedRef = React.useRef(false)

    const handlePickIngressFolder = React.useCallback(() => {
        pickRequestedRef.current = true
        onPickIngressFolder()
    }, [onPickIngressFolder])
    const steps: Step[] = React.useMemo(
        () => [
            {
                key: 'what',
                title: t('components.analyzerTutorialModal.steps.what.title'),
                body: (
                    <TextStack variant="support">
                        <Text as="p" variant="support">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.what.body"
                                components={{ b: <TextStrong /> }}
                            />
                        </Text>
                        <Text as="p" variant="support">
                            {t('components.analyzerTutorialModal.steps.what.bodyMuted')}
                        </Text>
                        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
                            <TextStack variant="support" className="space-y-1">
                                <Text as="div" variant="support">
                                    {t('components.analyzerTutorialModal.steps.what.footageNote')}
                                </Text>
                            </TextStack>
                        </div>
                    </TextStack>
                ),
            },
            {
                key: 'watch-folder',
                title: t('components.analyzerTutorialModal.steps.watchFolder.title'),
                body: (
                    <TextStack variant="support">
                        <Text as="p" variant="support">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.watchFolder.body"
                                components={{ b: <TextStrong /> }}
                            />
                        </Text>
                        <Text as="div" variant="support">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.watchFolder.safety"
                                components={{ b: <TextStrong /> }}
                            />
                        </Text>
                        <div className="rounded-xl border border-slate-200 bg-white p-3">
                            <Text as="div" variant="muted">
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.watchFolder.tip"
                                    components={{ b: <TextStrong /> }}
                                />
                            </Text>
                        </div>

                        <div className="rounded-xl border border-sky-200 bg-sky-50 p-3">
                            <Text as="div" variant="support">
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.watchFolder.uploadNotice"
                                    components={{ b: <TextStrong /> }}
                                />
                            </Text>
                        </div>

                        <div className="pt-2 flex flex-col items-center gap-2">
                            <Button
                                variant={ingressFolderName ? 'secondary' : 'primary'}
                                onClick={handlePickIngressFolder}
                                text={
                                    ingressFolderName
                                        ? t('common.actions.changeFolder')
                                        : t('components.analyzerTutorialModal.steps.watchFolder.button')
                                }
                            />
                            {ingressFolderName ? (
                                <div className="text-center">
                                    <Text as="div" variant="muted" weight="semibold" className="text-slate-900">
                                        {t('components.analyzerTutorialModal.steps.watchFolder.currentFolder', {
                                            name: ingressFolderName,
                                        })}
                                    </Text>
                                    <Text as="div" variant="muted">
                                        {t('components.analyzerTutorialModal.steps.watchFolder.status')}
                                    </Text>
                                </div>
                            ) : null}
                        </div>
                        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
                            <Text as="div" variant="support" className="text-amber-950">
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.watchFolder.permissions"
                                    components={{ b: <TextStrong /> }}
                                />
                            </Text>
                        </div>
                    </TextStack>
                ),
            },
            {
                key: 'add-videos',
                title: t('components.analyzerTutorialModal.steps.addVideos.title'),
                body: (
                    <TextStack variant="support">
                        {ingressFolderName ? (
                            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                                <Text as="div" variant="support" weight="semibold" className="text-slate-900">
                                    {t('components.analyzerTutorialModal.steps.addVideos.folderSelected', {
                                        name: ingressFolderName,
                                    })}
                                </Text>
                                <Text as="div" variant="muted" className="mt-0.5">
                                    {t('components.analyzerTutorialModal.steps.addVideos.statusWaiting')}
                                </Text>
                            </div>
                        ) : null}
                        <Text as="p" variant="support">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.addVideos.body"
                                components={{ b: <TextStrong /> }}
                            />
                        </Text>
                        <Text as="p" variant="support">
                            {t('components.analyzerTutorialModal.steps.addVideos.muted')}
                        </Text>
                        <Text as="div" variant="muted">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.addVideos.subfolders"
                                components={{ b: <TextStrong /> }}
                            />
                        </Text>
                    </TextStack>
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
                        <Text as="p" variant="muted">
                            {t('components.analyzerTutorialModal.steps.reviewRiding.tip')}
                        </Text>
                    </div>
                ),
            },
            {
                key: 'feedback-reports',
                title: t('components.analyzerTutorialModal.steps.feedbackReports.title'),
                body: (
                    <div className="space-y-3">
                        <Text as="p" variant="support">
                            {t('components.analyzerTutorialModal.steps.feedbackReports.body')}
                        </Text>
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
                        <Text as="p" variant="support">
                            {t('components.analyzerTutorialModal.steps.feedbackReports.muted')}
                        </Text>
                    </div>
                ),
            },
        ],
        [handlePickIngressFolder, ingressFolderName, t]
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
    const canClose = !isFirst
    const canGoNext = step.key !== 'watch-folder' || !!ingressFolderName
    const nextTooltip =
        step.key === 'watch-folder' && !ingressFolderName
            ? t('components.analyzerTutorialModal.steps.watchFolder.nextDisabledTooltip')
            : undefined
    const nextLabel =
        step.key === 'what'
            ? t('components.analyzerTutorialModal.actions.startSetup')
            : step.key === 'add-videos'
            ? t('components.analyzerTutorialModal.actions.startAddingVideos')
            : t('common.next')

    React.useEffect(() => {
        if (step.key !== 'watch-folder') return
        if (!pickRequestedRef.current) return
        if (!ingressFolderName) return
        pickRequestedRef.current = false
        const nextIdx = Math.min(visibleSteps.length - 1, safeIdx + 1)
        if (nextIdx === safeIdx) return
        onStepIndexChange(nextIdx)
    }, [ingressFolderName, onStepIndexChange, safeIdx, step.key, visibleSteps.length])

    // Set global style var(--z-ingress-widget) so the Video Folder widget is highlighted.
    if (step.key === 'watch-folder') {
        document.documentElement.style.setProperty('--z-ingress-widget', '1000')
    } else {
        document.documentElement.style.setProperty('--z-ingress-widget', 'auto')
    }

    return (
        <Modal
            onClose={canClose ? onClose : undefined}
            closeOnBackdropClick={canClose}
            closeOnEscape={canClose}
            showCloseButton={canClose}
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
                </div>
            }
        >
            <div className="p-4 max-h-[72vh] overflow-auto">
                <h3 className="m-0 text-lg font-semibold text-slate-900">{step.title}</h3>
                <div className="mt-3 text-sm text-slate-700 leading-6">{step.body}</div>
            </div>

            <div className="px-4 py-3 border-t border-slate-200 flex items-center justify-between gap-2">
                {isFirst ? (
                    <div />
                ) : (
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onStepIndexChange(Math.max(0, safeIdx - 1))}
                        text={t('common.back')}
                    />
                )}
                <div className="flex items-center gap-2">
                    {!isLast ? (
                        step.key === 'watch-folder' && !ingressFolderName ? null : nextTooltip ? (
                            <span title={nextTooltip} className="inline-flex cursor-not-allowed">
                                <Button
                                    variant="primary"
                                    size="sm"
                                    disabled
                                    className="pointer-events-none"
                                    text={t('common.next')}
                                />
                            </span>
                        ) : (
                            <Button
                                variant="primary"
                                size="sm"
                                onClick={() => onStepIndexChange(Math.min(visibleSteps.length - 1, safeIdx + 1))}
                                text={nextLabel}
                            />
                        )
                    ) : (
                        <Button
                            variant="primary"
                            size="sm"
                            disabled={!canClose}
                            onClick={canClose ? onClose : undefined}
                            text={t('common.done')}
                        />
                    )}
                </div>
            </div>
        </Modal>
    )
}
