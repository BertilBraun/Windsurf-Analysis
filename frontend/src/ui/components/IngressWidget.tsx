import React from 'react'
import { useIngressScanner } from '../hooks/useIngressScanner'
import type { IngressUploadItem, IngressUploadStatus } from '../hooks/useIngressScanner'
import type { UploadContext } from '../utils/uploader'
import { clamp } from '../utils/clamp'
import { Modal } from './Modal'

type Props = {
    dirHandle: FileSystemDirectoryHandle | null
    dirPermission: 'granted' | 'denied' | 'prompt' | null
    onPickDirectory: () => void
    uploadCtx: UploadContext
    onUploaded: () => void
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

export const IngressWidget: React.FC<Props> = ({ dirHandle, dirPermission, onPickDirectory, uploadCtx }) => {
    const scanner = useIngressScanner(dirHandle, uploadCtx)
    const [expanded, setExpanded] = React.useState(false)
    const [showQuotaModal, setShowQuotaModal] = React.useState(false)

    React.useEffect(() => {
        if (scanner.lastError?.toLowerCase().includes('quota')) setShowQuotaModal(true)
    }, [scanner.lastError])

    const uploading = scanner.uploading
    const ringPercent = meanProgress(scanner.uploads)
    const hasIssues = !dirHandle || dirPermission === 'denied' || !!scanner.lastError

    return (
        <>
            <div className="fixed bottom-4 right-4 z-50">
                {!expanded ? (
                    <button
                        type="button"
                        onClick={() => setExpanded(true)}
                        className={`flex items-center gap-2 rounded-full border px-3 py-2 shadow-sm bg-white hover:bg-slate-50 transition ${
                            hasIssues ? 'border-amber-300' : 'border-slate-200'
                        }`}
                        title="Ingress"
                        aria-label="Open ingress uploads"
                    >
                        {uploading > 0 ? <Ring percent={ringPercent} /> : <Ring percent={0} />}
                        <div className="flex flex-col items-start leading-tight">
                            <div className="text-xs font-semibold text-slate-900">Ingress</div>
                            <div className="text-[11px] text-slate-500">
                                {uploading > 0 ? `Uploading ${uploading}…` : dirHandle ? 'Monitoring' : 'Select folder'}
                            </div>
                        </div>
                        {uploading > 0 && (
                            <span className="ml-1 text-[11px] font-semibold text-brand-700 bg-brand-50 border border-brand-600/20 px-2 py-0.5 rounded-full">
                                {uploading}
                            </span>
                        )}
                    </button>
                ) : (
                    <div className="w-[380px] max-w-[92vw] rounded-2xl border border-slate-200 bg-white shadow-xl overflow-hidden">
                        <div className="px-4 py-3 border-b border-slate-200 flex items-center gap-3">
                            <div className="text-sm font-semibold">Ingress uploads</div>
                            <div className="flex-1" />
                            <button
                                type="button"
                                onClick={() => setExpanded(false)}
                                className="text-xs text-slate-600 hover:text-slate-900 px-2 py-1 rounded-md hover:bg-slate-100"
                            >
                                Close
                            </button>
                        </div>

                        <div className="p-4 flex flex-col gap-3">
                            <div className="text-xs text-slate-600">
                                Select your “windsurf analysis videos” folder. GybeLock monitors it and auto-uploads new
                                videos.
                            </div>

                            <div className="flex items-start justify-between gap-3">
                                <div className="text-xs text-slate-700">
                                    <div className="font-medium">Folder</div>
                                    {dirHandle ? (
                                        <div className="mt-0.5">
                                            <span className="text-slate-900">{dirHandle.name}</span>{' '}
                                            {dirPermission !== 'granted' ? (
                                                <span className="text-amber-700">(permission pending)</span>
                                            ) : null}
                                        </div>
                                    ) : (
                                        <div className="mt-0.5 text-amber-700">No folder selected</div>
                                    )}
                                </div>
                                <button
                                    type="button"
                                    className="px-3 py-2 rounded-md bg-brand-600 text-white text-sm hover:bg-brand-700 transition"
                                    onClick={onPickDirectory}
                                >
                                    {dirHandle ? 'Change folder' : 'Select folder'}
                                </button>
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
                                    <div className="text-xs font-semibold text-slate-900 mb-2">Uploads</div>
                                    <div className="flex flex-col gap-2 max-h-[40vh] overflow-auto pr-1">
                                        {scanner.uploads.map(item => (
                                            <UploadItem key={item.id} item={item} />
                                        ))}
                                    </div>
                                </div>
                            ) : (
                                <div className="text-xs text-slate-500">No uploads right now.</div>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {showQuotaModal && (
                <Modal onClose={() => setShowQuotaModal(false)} title="You're out of free jobs">
                    <div className="p-4 text-slate-900">
                        <p className="mb-3">
                            Hey, you've gotten a sneak peek of the Windsurf Analyzer. We'd love to get in contact to
                            hear your opinions — what you liked and what we could improve.
                        </p>
                        <p className="mb-1">
                            To get full and unlimited access to the analyzer, please reach out to us at
                            <br />
                            <a className="text-brand-700 underline" href="mailto:bertil.braun.private@gmail.com">
                                bertil.braun.private@gmail.com
                            </a>
                        </p>
                    </div>
                </Modal>
            )}
        </>
    )
}

const SuspendedBanner: React.FC<{ onRetryFailed: () => void }> = ({ onRetryFailed }) => {
    return (
        <div className="p-3 bg-amber-50 border border-amber-200 rounded-md">
            <div className="text-xs text-amber-800 mb-2">
                Uploads paused due to an error. You can retry failed uploads.
            </div>
            <div className="flex gap-2">
                <button
                    type="button"
                    className="px-3 py-2 rounded-md bg-slate-200 text-slate-800 text-sm hover:bg-slate-300 transition"
                    onClick={onRetryFailed}
                >
                    Retry failed
                </button>
            </div>
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
    const status = React.useMemo(() => {
        if (!dirHandle) return { text: 'Idle — no folder selected', color: '#b45309' }
        if (dirPermission === 'denied') return { text: 'Permission denied', color: '#b91c1c' }
        if (lastError) return { text: `Error — ${lastError}`, color: '#b91c1c' }
        if (uploading > 0)
            return { text: `Uploading ${uploading} file${uploading > 1 ? 's' : ''}…`, color: 'var(--brand-600)' }
        if (active)
            return {
                text: `Monitoring — last scan ${lastRunAt ? new Date(lastRunAt).toLocaleTimeString() : 'just now'}`,
                color: '#059669',
            }
        return { text: 'Idle', color: '#64748b' }
    }, [dirHandle, dirPermission, active, uploading, lastRunAt, lastError])

    return (
        <div className="flex items-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full" style={{ background: status.color }} />
            <span className="text-xs text-slate-700">Current status: {status.text}</span>
        </div>
    )
}

const UploadItem: React.FC<{ item: IngressUploadItem }> = ({ item }) => {
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
                    {isError ? 'Error' : isSkipped ? 'Skipped' : `${percent}%`}
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
