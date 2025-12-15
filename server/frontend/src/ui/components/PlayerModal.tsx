import React from 'react'
import { Modal } from './Modal'
import { CanvasPlayer } from '../player/CanvasPlayer'
import { JobDetail, ReportType } from '../types'
import { LogoButton } from './LogoButton'

export const PlayerModal: React.FC<{
    job: JobDetail
    dirHandle: FileSystemDirectoryHandle | null
    onClose: () => void
    onGoHome?: () => void
    onDelete: (id: string) => void
    onReport: (id: string, type: ReportType, message: string) => void
    onOpenNextJob?: () => void
    onOpenPrevJob?: () => void
    deletingId?: string | null
}> = ({ job, dirHandle, onClose, onGoHome, onDelete, onReport, onOpenNextJob, onOpenPrevJob, deletingId }) => {
    return (
        <>
            <Modal
                key={job.id}
                onClose={onClose}
                hideHeader
            >
                <div className="relative w-[96vw] h-[92vh] bg-white text-black rounded-2xl shadow-xl overflow-hidden border border-slate-200">
                    <div className="h-full w-full flex flex-col overflow-hidden">
                        <div className="h-12 flex items-center px-4 border-b border-slate-200 bg-white/80 backdrop-blur">
                            <LogoButton onClick={() => (onGoHome ? onGoHome() : onClose())} />
                        </div>
                        <CanvasPlayer
                            key={job.id}
                            job={job}
                            dirHandle={dirHandle}
                            onClose={onClose}
                            onDelete={onDelete}
                            onReport={onReport}
                            onOpenNextJob={onOpenNextJob}
                            onOpenPrevJob={onOpenPrevJob}
                        />
                    </div>
                </div>
            </Modal>
        </>
    )
}
