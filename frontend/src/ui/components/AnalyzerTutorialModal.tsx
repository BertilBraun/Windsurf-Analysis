import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { Modal } from './Modal'
import { Button } from './Button'
import { KeyboardShortcutsModal } from './KeyboardShortcutsModal'

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
    const [showShortcuts, setShowShortcuts] = React.useState(false)
    const steps: Step[] = React.useMemo(
        () => [
            {
                key: 'intro',
                title: t('components.analyzerTutorialModal.steps.intro.title'),
                body: (
                    <div className="space-y-3">
                        <p className="text-sm text-slate-700 leading-6">
                            {t('components.analyzerTutorialModal.steps.intro.lede')}
                        </p>
                        <div className="rounded-xl border border-slate-200 bg-white p-3">
                            <div className="text-sm font-semibold text-slate-900">
                                {t('components.analyzerTutorialModal.steps.intro.how.title')}
                            </div>
                            <ul className="mt-2 list-disc pl-5 space-y-1 text-sm text-slate-700">
                                <li>
                                    <Trans
                                        i18nKey="components.analyzerTutorialModal.steps.intro.how.bullets.ingress"
                                        components={{ b: <b /> }}
                                    />
                                </li>
                                <li>
                                    <Trans
                                        i18nKey="components.analyzerTutorialModal.steps.intro.how.bullets.upload"
                                        components={{ b: <b /> }}
                                    />
                                </li>
                                <li>
                                    <Trans
                                        i18nKey="components.analyzerTutorialModal.steps.intro.how.bullets.thumbnails"
                                        components={{ b: <b /> }}
                                    />
                                </li>
                            </ul>
                        </div>
                        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-amber-900">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.intro.note"
                                components={{ b: <b /> }}
                            />
                        </div>
                    </div>
                ),
            },
            {
                key: 'ingress-folder',
                title: t('components.analyzerTutorialModal.steps.ingress.title'),
                body: (
                    <div className="space-y-3">
                        <p className="text-sm text-slate-700 leading-6">
                            {t('components.analyzerTutorialModal.steps.ingress.lede')}
                        </p>
                        <ul className="list-disc pl-5 space-y-1">
                            <li>
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.ingress.bullets.select"
                                    components={{ b: <b /> }}
                                />
                            </li>
                            <li>{t('components.analyzerTutorialModal.steps.ingress.bullets.permission')}</li>
                        </ul>
                        <div className="text-xs text-slate-500">
                            <Trans
                                i18nKey="components.analyzerTutorialModal.steps.ingress.tip"
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
                                        : t('common.actions.selectFolder')
                                }
                            />
                            {ingressFolderName ? (
                                <div className="text-xs text-slate-600 text-center">
                                    <Trans
                                        i18nKey="components.analyzerTutorialModal.steps.ingress.current"
                                        components={{ strong: <span className="font-semibold text-slate-900" /> }}
                                        values={{ name: ingressFolderName }}
                                    />
                                </div>
                            ) : null}
                        </div>
                    </div>
                ),
            },
            {
                key: 'drop-mp4s',
                title: t('components.analyzerTutorialModal.steps.drop.title'),
                body: (
                    <div className="space-y-3">
                        <ul className="list-disc pl-5 space-y-1">
                            <li>
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.drop.bullets.copy"
                                    components={{
                                        code: <code className="px-1 py-0.5 rounded bg-slate-100" />,
                                        b: <b />,
                                    }}
                                />
                            </li>
                            <li>{t('components.analyzerTutorialModal.steps.drop.bullets.detect')}</li>
                            <li>
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.drop.bullets.ingress"
                                    components={{ b: <b /> }}
                                />
                            </li>
                        </ul>
                    </div>
                ),
            },
            {
                key: 'open-video',
                title: t('components.analyzerTutorialModal.steps.open.title'),
                body: (
                    <div className="space-y-3">
                        <ul className="list-disc pl-5 space-y-1">
                            <li>
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.open.bullets.thumbnail"
                                    components={{ b: <b /> }}
                                />
                            </li>
                            <li>{t('components.analyzerTutorialModal.steps.open.bullets.click')}</li>
                        </ul>
                    </div>
                ),
            },
            {
                key: 'review-track',
                title: t('components.analyzerTutorialModal.steps.review.title'),
                body: (
                    <div className="space-y-3">
                        <ul className="list-disc pl-5 space-y-1">
                            <li>
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.review.bullets.overview"
                                    components={{ b: <b /> }}
                                />
                            </li>
                            <li>{t('components.analyzerTutorialModal.steps.review.bullets.hover')}</li>
                            <li>{t('components.analyzerTutorialModal.steps.review.bullets.click')}</li>
                            <li>{t('components.analyzerTutorialModal.steps.review.bullets.timeline')}</li>
                        </ul>
                    </div>
                ),
            },
            {
                key: 'shortcuts-export-report',
                title: t('components.analyzerTutorialModal.steps.tips.title'),
                body: (
                    <div className="space-y-3">
                        <div className="rounded-xl border border-slate-200 bg-white p-3">
                            <div className="text-sm font-semibold text-slate-900">
                                {t('components.analyzerTutorialModal.steps.tips.shortcuts.title')}
                            </div>
                            <div className="mt-1 text-sm text-slate-700">
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.tips.shortcuts.body"
                                    components={{ b: <b /> }}
                                />
                            </div>
                            <div className="mt-3">
                                <Button
                                    variant="secondary"
                                    onClick={() => setShowShortcuts(true)}
                                    text={t('components.analyzerTutorialModal.steps.tips.shortcuts.button')}
                                />
                            </div>
                        </div>
                        <div className="rounded-xl border border-slate-200 bg-white p-3">
                            <div className="text-sm font-semibold text-slate-900">
                                {t('components.analyzerTutorialModal.steps.tips.export.title')}
                            </div>
                            <div className="mt-1 text-sm text-slate-700">
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.tips.export.body"
                                    components={{ b: <b /> }}
                                />
                            </div>
                        </div>
                        <div className="rounded-xl border border-slate-200 bg-white p-3">
                            <div className="text-sm font-semibold text-slate-900">
                                {t('components.analyzerTutorialModal.steps.tips.report.title')}
                            </div>
                            <div className="mt-1 text-sm text-slate-700">
                                <Trans
                                    i18nKey="components.analyzerTutorialModal.steps.tips.report.body"
                                    components={{ b: <b /> }}
                                />
                            </div>
                            <div className="mt-2 text-xs text-slate-500">
                                {t('components.analyzerTutorialModal.steps.tips.report.note')}
                            </div>
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

    // set global style var(z-ingress-widget) to 1000 so that the ingress widget is highlighted
    if (step.key === 'ingress-folder' || step.key === 'drop-mp4s') {
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

            {showShortcuts && <KeyboardShortcutsModal onClose={() => setShowShortcuts(false)} />}
        </>
    )
}
