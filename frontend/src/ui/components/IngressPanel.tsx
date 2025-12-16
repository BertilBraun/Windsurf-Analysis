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

export const IngressPanel: React.FC<Props> = ({ dirHandle, dirPermission, onPickDirectory, uploadCtx, onUploaded }) => {
    const scanner = useIngressScanner(dirHandle, uploadCtx, onUploaded)
    const [showQuotaModal, setShowQuotaModal] = React.useState(false)

    React.useEffect(() => {
        if (scanner.lastError?.toLowerCase().includes('quota')) setShowQuotaModal(true)
    }, [scanner.lastError])

    return (
        <div className="p-3 border border-gray-300 rounded-lg mb-4 bg-gray-50">
            <Header
                dirHandle={dirHandle}
                dirPermission={dirPermission}
                onPickDirectory={onPickDirectory}
                active={scanner.active}
                uploading={scanner.uploading}
                lastRunAt={scanner.lastRunAt}
                lastError={scanner.lastError}
            />

            {scanner.suspended && <SuspendedBanner onRetryFailed={scanner.retryFailed} />}

            {scanner.uploads.length > 0 && <UploadsList items={scanner.uploads} />}

            {showQuotaModal && (
                <Modal onClose={() => setShowQuotaModal(false)} title="You're out of free jobs">
                    <div className="p-4 text-gray-100">
                        <p className="mb-3">
                            Hey, you've gotten a sneak peek of the Windsurf Analyzer. We'd love to get in contact to
                            hear your opinions — what you liked and what we could improve.
                        </p>
                        <p className="mb-3">
                            To get full and unlimited access to the analyzer, please reach out to us at
                            <br />
                            <a className="text-blue-400 underline" href="mailto:bertil.braun.private@gmail.com">
                                bertil.braun.private@gmail.com
                            </a>
                        </p>
                    </div>
                </Modal>
            )}
        </div>
    )
}

type HeaderProps = {
    dirHandle: FileSystemDirectoryHandle | null
    dirPermission: 'granted' | 'denied' | 'prompt' | null
    onPickDirectory: () => void
    active: boolean
    uploading: number
    lastRunAt: number | null
    lastError: string | null
}

const Header: React.FC<HeaderProps> = ({
    dirHandle,
    dirPermission,
    onPickDirectory,
    active,
    uploading,
    lastRunAt,
    lastError,
}) => {
    return (
        <div className="flex justify-between items-center gap-3">
            <div className="flex flex-col flex-1">
                <strong>Ingress folder</strong>
                <span className="text-xs text-gray-500">
                    Select your "windsurf analysis videos" folder. We'll monitor it every 10s and auto-upload new
                    videos.
                </span>
                {dirHandle ? (
                    <span className="text-xs text-gray-700 mt-1">
                        Selected: {dirHandle.name} {dirPermission !== 'granted' ? '(permission pending)' : ''}
                    </span>
                ) : (
                    <span className="text-xs text-red-700 mt-1">No folder selected</span>
                )}
                <StatusLine
                    dirHandle={dirHandle}
                    dirPermission={dirPermission}
                    active={active}
                    uploading={uploading}
                    lastRunAt={lastRunAt}
                    lastError={lastError}
                />
            </div>
            <div className="flex gap-2">
                <button
                    className="px-3 py-1.5 rounded-md bg-blue-600 text-white text-sm hover:bg-blue-700 active:bg-blue-800 disabled:opacity-60"
                    onClick={onPickDirectory}
                >
                    {dirHandle ? 'Change folder' : 'Select folder'}
                </button>
            </div>
        </div>
    )
}

const SuspendedBanner: React.FC<{ onRetryFailed: () => void }> = ({ onRetryFailed }) => {
    return (
        <div className="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-md">
            <div className="text-xs text-amber-800 mb-2">
                Uploads paused due to an error. You can retry failed uploads.
            </div>
            <div className="flex gap-2">
                <button
                    className="px-3 py-1.5 rounded-md bg-gray-200 text-gray-800 text-sm hover:bg-gray-300 active:bg-gray-400"
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
        if (!dirHandle) return { text: 'Idle — no folder selected', color: '#b91c1c' }
        if (dirPermission === 'denied') return { text: 'Permission denied', color: '#b91c1c' }
        if (lastError) return { text: `Error — ${lastError}`, color: '#b91c1c' }
        if (uploading > 0) return { text: `Uploading ${uploading} file${uploading > 1 ? 's' : ''}…`, color: '#2563eb' }
        if (active)
            return {
                text: `Monitoring — last scan ${lastRunAt ? new Date(lastRunAt).toLocaleTimeString() : 'just now'}`,
                color: '#059669',
            }
        return { text: 'Idle', color: '#6b7280' }
    }, [dirHandle, dirPermission, active, uploading, lastRunAt, lastError])

    return (
        <div className="flex items-center gap-2 mt-1">
            <span className="inline-block w-2 h-2 rounded-full" style={{ background: status.color }} />
            <span className="text-xs text-gray-700">Current status: {status.text}</span>
        </div>
    )
}

const UploadsList: React.FC<{ items: IngressUploadItem[] }> = ({ items }) => {
    return (
        <div className="mt-3">
            <strong className="text-sm block mb-2">Uploads</strong>
            <div className="flex flex-col gap-2">
                {items.map(item => (
                    <UploadItem key={item.id} item={item} />
                ))}
            </div>
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
                        isError ? 'text-red-700' : 'text-gray-700'
                    } overflow-hidden text-ellipsis whitespace-nowrap flex-1`}
                    title={item.relativePath}
                >
                    {item.relativePath}
                </span>
                <span className="text-xs text-gray-500 min-w-[36px] text-right">
                    {isError ? 'Error' : isSkipped ? 'Skipped Duplicate' : `${percent}%`}
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
            ? 'bg-blue-400'
            : status === 'done'
            ? 'bg-green-400'
            : 'bg-gray-400'
    return (
        <div className="relative h-1.5 bg-gray-200 rounded-full overflow-hidden">
            <div className={`absolute left-0 top-0 bottom-0 ${color}`} style={{ width: `${percent}%` }} />
        </div>
    )
}
