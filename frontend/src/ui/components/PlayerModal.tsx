import React from 'react'
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
                title={job.local_relative_path?.replace(/\.mp4$/i, '') ?? 'n/a'}
                additionalHeader={
                    <>
                        <Button
                            onClick={toggleDrawMode}
                            title="Toggle draw mode (D)"
                            text="Draw"
                            variant={drawMode ? 'primary' : 'secondary'}
                        />
                        <Button onClick={() => setShowShortcuts(true)} title="Keyboard shortcuts" text="Shortcuts" />
                        <Button
                            onClick={() => setShowReport(true)}
                            title="Report an issue with this analysis"
                            text="Report"
                        />
                        <Button
                            onClick={() => onDelete(job.id)}
                            title="Delete job"
                            text="Delete"
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
    const [type, setType] = React.useState<ReportType>('missed_detection')
    const [message, setMessage] = React.useState<string>('')
    const [isSubmitting, setIsSubmitting] = React.useState<boolean>(false)
    const [error, setError] = React.useState<string | null>(null)

    const canSubmit = !isSubmitting && message.trim().length > 0

    return (
        <Modal onClose={onClose} title="Report an issue">
            <div className="p-4 space-y-4 max-w-[720px]">
                <div className="text-sm text-slate-600">
                    Help us improve the analysis quality for{' '}
                    <span className="font-medium text-slate-800">{job.local_relative_path ?? job.id}</span>.
                </div>

                <div>
                    <label className="block text-sm font-medium text-slate-900 mb-1">Issue type</label>
                    <select
                        className="w-full bg-white border border-slate-200 rounded-md p-2 text-slate-900"
                        value={type}
                        onChange={e => setType(e.target.value as ReportType)}
                        disabled={isSubmitting}
                    >
                        <option value="missed_detection">Missed detection</option>
                        <option value="false_association">False association</option>
                        <option value="other">Other</option>
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-slate-900 mb-1">Describe - What went wrong?</label>
                    <textarea
                        className="w-full min-h-28 bg-white border border-slate-200 rounded-md p-2 text-slate-900"
                        placeholder="Example: Rider missed around 00:12–00:18, or track switches riders at ~00:43."
                        value={message}
                        onChange={e => setMessage(e.target.value)}
                        disabled={isSubmitting}
                    />
                    <div className="mt-2 text-xs text-slate-500">Tip: include an approximate timestamp if you can.</div>
                </div>

                {error && <div className="text-sm text-red-700">{error}</div>}

                <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-200">
                    <Button variant="ghost" onClick={onClose} text="Cancel" disabled={isSubmitting} />
                    <Button
                        variant="primary"
                        text={isSubmitting ? 'Sending report…' : 'Send report'}
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
