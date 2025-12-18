import React from 'react'
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
}

export const AnalyzerTutorialModal: React.FC<AnalyzerTutorialModalProps> = ({
    onClose,
    stepIndex,
    onStepIndexChange,
    onPickIngressFolder,
    ingressFolderName,
}) => {
    const [showShortcuts, setShowShortcuts] = React.useState(false)
    const steps: Step[] = React.useMemo(
        () => [
            {
                key: 'intro',
                title: 'What GybeLock does (and what footage works)',
                body: (
                    <div className="space-y-3">
                        <p className="text-sm text-slate-700 leading-6">
                            GybeLock turns shaky, long-zoom beach videos into a smooth player where you can click a
                            rider and review a focused (cropped) track.
                        </p>
                        <div className="rounded-xl border border-slate-200 bg-white p-3">
                            <div className="text-sm font-semibold text-slate-900">How it works (high level)</div>
                            <ul className="mt-2 list-disc pl-5 space-y-1 text-sm text-slate-700">
                                <li>
                                    You choose an <b>ingress folder</b> on your computer.
                                </li>
                                <li>
                                    GybeLock monitors it and <b>auto-uploads</b> new MP4s you drop in.
                                </li>
                                <li>
                                    Processed videos then appear in <b>Analyzed Videos</b> as thumbnails you can open.
                                </li>
                            </ul>
                        </div>
                        <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-amber-900">
                            Designed for <b>beach-shot</b> footage (shore camera, long zoom/tele). It is <b>not</b>{' '}
                            designed for GoPro/action-cam POV footage.
                        </div>
                    </div>
                ),
            },
            {
                key: 'ingress-folder',
                title: '1) Set the ingress folder',
                body: (
                    <div className="space-y-3">
                        <p className="text-sm text-slate-700 leading-6">
                            First, select a folder on your computer where you want to drop your videos. GybeLock will
                            monitor that folder and automatically upload new MP4s you add there.
                        </p>
                        <ul className="list-disc pl-5 space-y-1">
                            <li>
                                Either here or in Ingress (bottom-right), click <b>Select folder</b> and choose where
                                you’ll drop your beach-shot MP4 videos.
                            </li>
                            <li>You will need to grant folder permission so GybeLock can keep monitoring it.</li>
                        </ul>
                        <div className="text-xs text-slate-500">
                            Tip: After selecting a folder, check that it shows <b>Monitoring</b> (or uploading if you
                            just added videos).
                        </div>

                        <div className="pt-2 flex flex-col items-center gap-2">
                            <Button
                                variant={ingressFolderName ? 'secondary' : 'primary'}
                                onClick={() => onPickIngressFolder()}
                                text={ingressFolderName ? 'Change folder' : 'Select folder'}
                            />
                            {ingressFolderName ? (
                                <div className="text-xs text-slate-600 text-center">
                                    Current: <span className="font-semibold text-slate-900">{ingressFolderName}</span>
                                </div>
                            ) : null}
                        </div>
                    </div>
                ),
            },
            {
                key: 'drop-mp4s',
                title: '2) Drop MP4s into the folder',
                body: (
                    <div className="space-y-3">
                        <ul className="list-disc pl-5 space-y-1">
                            <li>
                                Copy or move your <code className="px-1 py-0.5 rounded bg-slate-100">.mp4</code> files
                                into the ingress folder (<b>subfolders are fine</b>).
                            </li>
                            <li>GybeLock will automatically detect new files and start uploading.</li>
                            <li>
                                Open <b>Ingress</b> anytime to see upload progress and errors.
                            </li>
                        </ul>
                    </div>
                ),
            },
            {
                key: 'open-video',
                title: '3) Open an analyzed video',
                body: (
                    <div className="space-y-3">
                        <ul className="list-disc pl-5 space-y-1">
                            <li>
                                In <b>Analyzed Videos</b>, look for a tile that shows a <b>video thumbnail</b>.
                            </li>
                            <li>Click the tile to open the player (tiles become clickable once ready).</li>
                        </ul>
                        <details className="rounded-xl border border-amber-200 bg-amber-50 p-3">
                            <summary className="cursor-pointer text-sm font-semibold text-amber-900">
                                Troubleshooting: “VIDEO FILE NOT FOUND”
                            </summary>
                            <div className="mt-2 text-sm text-amber-900/90">
                                <ul className="list-disc pl-5 space-y-1">
                                    <li>Re-check you selected the correct ingress folder</li>
                                    <li>Make sure the video still exists at its expected path</li>
                                </ul>
                            </div>
                        </details>
                    </div>
                ),
            },
            {
                key: 'review-track',
                title: '4) Review a track',
                body: (
                    <div className="space-y-3">
                        <ul className="list-disc pl-5 space-y-1">
                            <li>
                                In the player, you start in <b>overview</b> mode.
                            </li>
                            <li>Move your mouse over a rider to highlight the detection.</li>
                            <li>Click a rider to switch into a focused (cropped) view for that track.</li>
                            <li>Use the timeline to seek.</li>
                        </ul>
                    </div>
                ),
            },
            {
                key: 'shortcuts-export-report',
                title: 'Tips: shortcuts, export, report',
                body: (
                    <div className="space-y-3">
                        <div className="rounded-xl border border-slate-200 bg-white p-3">
                            <div className="text-sm font-semibold text-slate-900">Shortcuts (highly recommended)</div>
                            <div className="mt-1 text-sm text-slate-700">
                                There are many shortcuts (seeking, frame stepping, track navigation, etc.) that make
                                reviewing a session much faster. In the <b>player</b>, click <b>Shortcuts</b> to see all
                                keyboard controls.
                            </div>
                            <div className="mt-3">
                                <Button
                                    variant="secondary"
                                    onClick={() => setShowShortcuts(true)}
                                    text="Show shortcuts"
                                />
                            </div>
                        </div>
                        <div className="rounded-xl border border-slate-200 bg-white p-3">
                            <div className="text-sm font-semibold text-slate-900">Export</div>
                            <div className="mt-1 text-sm text-slate-700">
                                In the player, <b>Export</b> saves the <b>currently selected track</b> as an MP4 so you
                                can keep it and share it with others.
                            </div>
                        </div>
                        <div className="rounded-xl border border-slate-200 bg-white p-3">
                            <div className="text-sm font-semibold text-slate-900">Report</div>
                            <div className="mt-1 text-sm text-slate-700">
                                If you notice issues with <b>detections</b> (a rider wasn’t clickable) or{' '}
                                <b>tracking</b> (a focused track switches riders, or one rider ends up split into two
                                tracks), please use <b>Report</b> in the player.
                            </div>
                            <div className="mt-2 text-xs text-slate-500">
                                We’re still in beta — these failure cases are invaluable for improving the service. If
                                possible, include an approximate timestamp.
                            </div>
                        </div>
                    </div>
                ),
            },
        ],
        [ingressFolderName, onPickIngressFolder]
    )

    // Step state is controlled by the parent (session-only, no persistence).
    // This lets users close/reopen the modal and continue where they left off,
    // but a page refresh will reset back to step 1.
    const safeIdx = Math.min(Math.max(0, stepIndex || 0), steps.length - 1)
    const step = steps[safeIdx] ?? steps[0]!
    const isFirst = safeIdx === 0
    const isLast = safeIdx === steps.length - 1

    return (
        <>
            <Modal
                onClose={onClose}
                title="Analyzer Tutorial"
                contentClassName="rounded-2xl border border-slate-200 bg-white shadow-xl w-[760px] max-w-[96vw]"
                additionalHeader={
                    <div className="flex items-center gap-2">
                        <div className="text-xs text-slate-600">
                            Step <span className="font-semibold">{safeIdx + 1}</span> of{' '}
                            <span className="font-semibold">{steps.length}</span>
                        </div>
                        <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => onStepIndexChange(0)}
                            text="Restart"
                            title="Restart tutorial"
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
                        text="Back"
                    />
                    <div className="flex items-center gap-2">
                        {!isLast ? (
                            <Button
                                variant="primary"
                                size="sm"
                                onClick={() => onStepIndexChange(Math.min(steps.length - 1, safeIdx + 1))}
                                text="Next"
                            />
                        ) : (
                            <Button variant="primary" size="sm" onClick={onClose} text="Done" />
                        )}
                    </div>
                </div>
            </Modal>

            {showShortcuts && <KeyboardShortcutsModal onClose={() => setShowShortcuts(false)} />}
        </>
    )
}
