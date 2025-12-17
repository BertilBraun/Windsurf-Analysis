import React from 'react'
import { Modal } from './Modal'
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
}> = ({ job, dirHandle, onClose, onDelete, onReport, onOpenNextJob, onOpenPrevJob }) => {
    return (
        <>
            <Modal key={job.id} onClose={onClose}>
                <div className="relative w-[96vw] h-[92vh] bg-white text-black rounded-2xl shadow-xl overflow-hidden border border-slate-200">
                    <div className="h-full w-full flex flex-col overflow-hidden">
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
