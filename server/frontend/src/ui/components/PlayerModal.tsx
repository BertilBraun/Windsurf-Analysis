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
    deletingId?: string | null
}> = ({ job, dirHandle, onClose, onDelete, onReport, deletingId }) => {
    const [showShortcuts, setShowShortcuts] = React.useState<boolean>(false)

    return (
        <>
            <Modal
                onClose={onClose}
                title={job.original_file_path}
                additionalHeader={
                    <>
                        <Button onClick={() => setShowShortcuts(true)} title="Keyboard shortcuts" text="Shortcuts" />
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
                            job={job}
                            dirHandle={dirHandle}
                            onClose={onClose}
                            onDelete={onDelete}
                            onReport={onReport}
                        />
                    </div>
                </div>
            </Modal>

            {showShortcuts && <KeyboardShortcutsModal onClose={() => setShowShortcuts(false)} />}
        </>
    )
}
