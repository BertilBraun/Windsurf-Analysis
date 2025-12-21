import React from 'react'
import { useTranslation } from 'react-i18next'
import { useIngressScanner } from '../hooks/useIngressScanner'
import type { IngressUploadItem, IngressUploadStatus } from '../hooks/useIngressScanner'
import type { UploadContext } from '../utils/uploader'
import { clamp } from '../utils/clamp'
import { Modal } from './Modal'
import { Button } from './Button'

type Props = {
    dirHandle: FileSystemDirectoryHandle | null
    dirPermission: 'granted' | 'denied' | 'prompt' | null
    onPickDirectory: () => void
    uploadCtx: UploadContext
    knownChecksumsSha256?: ReadonlySet<string> | null
    enabled?: boolean
}

function meanProgress(items: IngressUploadItem[]) {
    const uploading = items.filter(i => i.status === 'uploading')
    if (uploading.length === 0) return 0
    const total = uploading.reduce((sum, i) => sum + clamp(i.progress, 0, 100), 0)
    return Math.round(total / uploading.length)
}

const Ring: React.FC<{ percent: number }> = ({ percent }) => {
    const r = 10
    const c = 2 * Math.PI * r
    const p = clamp(percent, 0, 100)
    const dash = (p / 100) * c
    return (
        <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
            <circle cx="14" cy="14" r={r} fill="none" stroke="rgba(15,23,42,0.12)" strokeWidth="3" />
            <circle
                cx="14"
                cy="14"
                r={r}
                fill="none"
                stroke="var(--brand-600)"
                strokeWidth="3"
                strokeLinecap="round"
                strokeDasharray={`${dash} ${c - dash}`}
                transform="rotate(-90 14 14)"
            />
        </svg>
    )
}

export const IngressWidget: React.FC<Props> = ({
    dirHandle,
    dirPermission,
    onPickDirectory,
    uploadCtx,
    knownChecksumsSha256 = null,
    enabled = true,
}) => {
    const { t } = useTranslation()
    const scanner = useIngressScanner(dirHandle, uploadCtx, knownChecksumsSha256, enabled)
    const [expanded, setExpanded] = React.useState(false)
    const [showQuotaModal, setShowQuotaModal] = React.useState(false)
    const panelRef = React.useRef<HTMLDivElement | null>(null)

    React.useEffect(() => {
        if (scanner.lastError?.toLowerCase().includes('quota')) setShowQuotaModal(true)
    }, [scanner.lastError])

    React.useEffect(() => {
        if (!expanded) return
        const onPointerDown = (event: MouseEvent) => {
            const target = event.target as Node | null
            if (!target) return
            if (panelRef.current?.contains(target)) return
            setExpanded(false)
        }
        document.addEventListener('mousedown', onPointerDown)
        return () => document.removeEventListener('mousedown', onPointerDown)
    }, [expanded])

    React.useEffect(() => {
        if (scanner.uploading <= 0) return
        const warning = t('components.ingressWidget.leaveWarning')
        const onBeforeUnload = (event: BeforeUnloadEvent) => {
            event.preventDefault()
            event.returnValue = warning
            return warning
        }
        window.addEventListener('beforeunload', onBeforeUnload)
        return () => window.removeEventListener('beforeunload', onBeforeUnload)
    }, [scanner.uploading, t])

    const uploading = scanner.uploading
    const ringPercent = meanProgress(scanner.uploads)
    const hasIssues = !dirHandle || dirPermission !== 'granted' || !!scanner.lastError

    const issue = React.useMemo(() => {
        if (!dirHandle) {
            return {
                kind: 'warning' as const,
                title: t('components.ingressWidget.issue.noFolder.title'),
                message: t('components.ingressWidget.issue.noFolder.message'),
            }
        }
        if (dirPermission === 'denied') {
            return {
                kind: 'error' as const,
                title: t('components.ingressWidget.issue.permissionDenied.title'),
                message: t('components.ingressWidget.issue.permissionDenied.message'),
            }
        }
        if (scanner.lastError) {
            return {
                kind: 'error' as const,
                title: t('components.ingressWidget.issue.ingressError.title'),
                message: scanner.lastError,
            }
        }
        if (dirPermission && dirPermission !== 'granted') {
            return {
                kind: 'warning' as const,
                title: t('components.ingressWidget.issue.permissionNeeded.title'),
                message: t('components.ingressWidget.issue.permissionNeeded.message'),
            }
        }
        return null
    }, [dirHandle, dirPermission, scanner.lastError, t])

    const fabStatus = React.useMemo(() => {
        if (uploading > 0) return t('components.ingressWidget.fab.status.uploading', { count: uploading })
        if (issue?.kind === 'error') return t('components.ingressWidget.fab.status.error')
        if (dirHandle) return t('components.ingressWidget.fab.status.monitoring')
        return t('components.ingressWidget.fab.status.selectFolder')
    }, [dirHandle, issue?.kind, t, uploading])

    return (
        <>
            <div className="fixed bottom-4 right-4 z-50">
                {!expanded ? (
                    <Button
                        type="button"
                        variant="unstyled"
                        size="none"
                        onClick={() => setExpanded(true)}
                        className={`flex items-center gap-2 rounded-full border px-3 py-2 shadow-sm transition ${
                            hasIssues
                                ? 'border-amber-500 bg-amber-100 hover:bg-amber-200'
                                : 'border-slate-200 bg-white hover:bg-slate-50'
                        }`}
                        title={t('components.ingressWidget.fab.title')}
                        aria-label={t('components.ingressWidget.fab.aria')}
                    >
                        {uploading > 0 ? <Ring percent={ringPercent} /> : <Ring percent={0} />}
                        <div className="flex flex-col items-start leading-tight">
                            <div className="text-xs font-semibold text-slate-900">
                                {t('components.ingressWidget.fab.title')}
                            </div>
                            <div className="text-[11px] text-slate-500">{fabStatus}</div>
                        </div>
                        {uploading > 0 && (
                            <span className="ml-1 text-[11px] font-semibold text-brand-700 bg-brand-50 border border-brand-600/20 px-2 py-0.5 rounded-full">
                                {uploading}
                            </span>
                        )}
                    </Button>
                ) : (
                    <div
                        ref={panelRef}
                        className={`w-[380px] max-w-[92vw] rounded-2xl border bg-white shadow-xl overflow-hidden ${
                            issue?.kind === 'error'
                                ? 'border-red-300'
                                : issue?.kind === 'warning'
                                ? 'border-amber-300'
                                : 'border-slate-200'
                        }`}
                    >
                        <div className="px-4 py-3 border-b border-slate-200 flex items-center gap-3">
                            <div className="text-sm font-semibold">{t('components.ingressWidget.panel.title')}</div>
                            <div className="flex-1" />
                            <Button
                                type="button"
                                variant="unstyled"
                                size="none"
                                onClick={() => setExpanded(false)}
                                className="text-xs text-slate-600 hover:text-slate-900 px-2 py-1 rounded-md hover:bg-slate-100"
                                text={t('common.close')}
                            />
                        </div>

                        <div className="p-4 flex flex-col gap-3">
                            {issue && (
                                <IssueBanner kind={issue.kind} title={issue.title}>
                                    {issue.message}
                                </IssueBanner>
                            )}

                            <div className="text-xs text-slate-600">{t('components.ingressWidget.panel.description')}</div>

                            <div className="flex items-start justify-between gap-3">
                                <div className="text-xs text-slate-700">
                                    <div className="font-medium">{t('components.ingressWidget.panel.folderLabel')}</div>
                                    {dirHandle ? (
                                        <div className="mt-0.5">
                                            <span className="text-slate-900">{dirHandle.name}</span>{' '}
                                            {dirPermission !== 'granted' ? (
                                                <span className="text-amber-700">
                                                    {t('components.ingressWidget.panel.permissionPending')}
                                                </span>
                                            ) : null}
                                        </div>
                                    ) : (
                                        <div className="mt-0.5 font-semibold text-amber-800">
                                            {t('components.ingressWidget.panel.noFolder')}
                                        </div>
                                    )}
                                </div>
                                <Button
                                    type="button"
                                    variant="unstyled"
                                    size="none"
                                    className={`px-3 py-2 rounded-md text-white text-sm transition ${
                                        !dirHandle
                                            ? 'bg-amber-500 hover:bg-amber-600'
                                            : 'bg-brand-600 hover:bg-brand-700'
                                    }`}
                                    onClick={onPickDirectory}
                                    text={dirHandle ? t('common.actions.changeFolder') : t('common.actions.selectFolder')}
                                />
                            </div>

                            <StatusLine
                                dirHandle={dirHandle}
                                dirPermission={dirPermission}
                                active={scanner.active}
                                uploading={scanner.uploading}
                                lastRunAt={scanner.lastRunAt}
                                lastError={scanner.lastError}
                            />

                            {scanner.suspended && <SuspendedBanner onRetryFailed={scanner.retryFailed} />}

                            {scanner.uploads.length > 0 ? (
                                <div className="mt-1">
                                    <div className="text-xs font-semibold text-slate-900 mb-2">
                                        {t('components.ingressWidget.panel.uploadsTitle')}
                                    </div>
                                    <div className="flex flex-col gap-2 max-h-[40vh] overflow-auto pr-1">
                                        {scanner.uploads.map(item => (
                                            <UploadItem key={item.id} item={item} />
                                        ))}
                                    </div>
                                </div>
                            ) : (
                                <div className="text-xs text-slate-500">{t('components.ingressWidget.panel.noUploads')}</div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {showQuotaModal && (
                <Modal onClose={() => setShowQuotaModal(false)} title={t('components.ingressWidget.quota.title')}>
                    <div className="p-4 text-slate-900">
                        <p className="mb-3">{t('components.ingressWidget.quota.body1')}</p>
                        <p className="mb-1">
                            {t('components.ingressWidget.quota.body2')}
                            <br />
                            <a className="text-brand-700 underline" href="mailto:contact@gybelock.de">
                                contact@gybelock.de
                            </a>
                        </p>
                    </div>
                </Modal>
            )}
        </>
    )
}

const SuspendedBanner: React.FC<{ onRetryFailed: () => void }> = ({ onRetryFailed }) => {
    const { t } = useTranslation()
    return (
        <div className="p-3 bg-amber-50 border border-amber-200 rounded-md">
            <div className="text-xs text-amber-800 mb-2">{t('components.ingressWidget.suspended.message')}</div>
            <div className="flex gap-2">
                <Button
                    type="button"
                    variant="unstyled"
                    size="none"
                    className="px-3 py-2 rounded-md bg-slate-200 text-slate-800 text-sm hover:bg-slate-300 transition"
                    onClick={onRetryFailed}
                    text={t('components.ingressWidget.suspended.retry')}
                />
            </div>
        </div>
    )
}

const IssueBanner: React.FC<{ kind: 'warning' | 'error'; title: string; children: React.ReactNode }> = ({
    kind,
    title,
    children,
}) => {
    const styles =
        kind === 'error'
            ? 'bg-red-50 border-red-300 text-red-900'
            : kind === 'warning'
            ? 'bg-amber-50 border-amber-300 text-amber-900'
            : 'bg-slate-50 border-slate-200 text-slate-900'

    const titleStyles = kind === 'error' ? 'text-red-900' : 'text-amber-900'

    return (
        <div className={`p-3 border rounded-md ${styles}`} role="alert" aria-live="polite">
            <div className={`text-xs font-semibold ${titleStyles}`}>{title}</div>
            <div className="text-xs mt-1 opacity-90">{children}</div>
        </div>
    )
}

type StatusLineProps = {
    dirHandle: FileSystemDirectoryHandle | null
    dirPermission: 'granted' | 'denied' | 'prompt' | null
    active: boolean
    uploading: number
    lastRunAt: number | null
    lastError: string | null
}

const StatusLine: React.FC<StatusLineProps> = ({
    dirHandle,
    dirPermission,
    active,
    uploading,
    lastRunAt,
    lastError,
}) => {
    const { t } = useTranslation()
    const status = React.useMemo(() => {
        if (!dirHandle) return { text: t('components.ingressWidget.statusLine.idleNoFolder'), color: '#b45309' }
        if (dirPermission === 'denied')
            return { text: t('components.ingressWidget.statusLine.permissionDenied'), color: '#b91c1c' }
        if (lastError)
            return { text: t('components.ingressWidget.statusLine.error', { message: lastError }), color: '#b91c1c' }
        if (uploading > 0)
            return {
                text: t('components.ingressWidget.statusLine.uploading', { count: uploading }),
                color: 'var(--brand-600)',
            }
        if (active) {
            const lastScanLabel = lastRunAt
                ? new Date(lastRunAt).toLocaleTimeString()
                : t('components.ingressWidget.statusLine.justNow')
            return {
                text: t('components.ingressWidget.statusLine.monitoring', { time: lastScanLabel }),
                color: '#059669',
            }
        }
        return { text: t('components.ingressWidget.statusLine.idle'), color: '#64748b' }
    }, [active, dirHandle, dirPermission, lastError, lastRunAt, t, uploading])

    return (
        <div className="flex items-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full" style={{ background: status.color }} />
            <span className="text-xs text-slate-700">
                {t('components.ingressWidget.statusLine.label', { status: status.text })}
            </span>
        </div>
    )
}

const UploadItem: React.FC<{ item: IngressUploadItem }> = ({ item }) => {
    const { t } = useTranslation()
    const percent = clamp(item.progress, 0, 100)
    const isError = item.status === 'error'
    const isSkipped = item.status === 'skipped'
    return (
        <div className="flex flex-col gap-1">
            <div className="flex justify-between gap-2">
                <span
                    className={`text-xs ${
                        isError ? 'text-red-700' : 'text-slate-700'
                    } overflow-hidden text-ellipsis whitespace-nowrap flex-1`}
                    title={item.relativePath}
                >
                    {item.relativePath}
                </span>
                <span className="text-xs text-slate-500 min-w-[36px] text-right">
                    {isError
                        ? t('components.ingressWidget.uploadItem.status.error')
                        : isSkipped
                        ? t('components.ingressWidget.uploadItem.status.skipped')
                        : `${percent}%`}
                </span>
            </div>
            <ProgressBar percent={percent} status={item.status} />
        </div>
    )
}

const ProgressBar: React.FC<{
    percent: number
    status: IngressUploadStatus
}> = ({ percent, status }) => {
    const color =
        status === 'error'
            ? 'bg-red-300'
            : status === 'uploading'
            ? 'bg-green-500'
            : status === 'done'
            ? 'bg-green-500'
            : status === 'skipped'
            ? 'bg-slate-400'
            : 'bg-slate-400'
    return (
        <div className="relative h-1.5 bg-slate-200 rounded-full overflow-hidden">
            <div
                className={`absolute left-0 top-0 bottom-0 ${color} transition-[width] duration-150 ease-linear`}
                style={{ width: `${percent}%` }}
            />
        </div>
    )
}
