/**
 * @file DemoPage.tsx
 * @description Demo interface for single-video analysis, allowing users to test
 * platform capabilities with local files or sample videos.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../auth/AuthProvider'
import { Button } from '../components/Button'
import JobThumbnail from '../components/JobThumbnail'
import { LogoButton } from '../components/LogoButton'
import { PlayerModal } from '../components/PlayerModal'
import { useJobs } from '../hooks/useJobs'
import { computeSha256 } from '../utils/localFileIndex'
import { uploadVideoFile } from '../utils/uploader'
import type { JobDetail, JobSummary } from '../types'
import { useNavigate } from 'react-router-dom'

type DemoItemPhase = 'queued' | 'hashing' | 'uploading' | 'waiting' | 'error'

type DemoItem = {
    id: string
    createdAtMs: number
    file: File
    sha256: string | null
    jobId: string | null
    progressPct: number | null
    phase: DemoItemPhase
    error: string | null
    duplicateSkipped: boolean
}

const DEMO_CONCURRENT_WORKERS = 2

function isTerminal(status: JobSummary['status']): boolean {
    return status === 'succeeded' || status === 'failed' || status === 'canceled'
}

const UploadProgressBar: React.FC<{ percent: number }> = ({ percent }) => {
    const p = Math.max(0, Math.min(100, percent))
    return (
        <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={p}>
            <div className="h-full bg-brand-600 transition-[width] duration-150" style={{ width: `${p}%` }} />
        </div>
    )
}

/**
 * A simplified interface for demonstrating video analysis capabilities.
 *
 * Features include:
 * - Single file upload with progress tracking.
 * - Sample video processing.
 * - Automatic playback upon successful analysis.
 * - Navigation to the full Analyzer application.
 *
 * @param props - Component properties.
 * @param props.onGoHome - Callback triggered when the user requests to return to the home screen.
 */
export const DemoPage: React.FC<{ onGoHome: () => void }> = ({ onGoHome }) => {
    const { t } = useTranslation()
    const { authorizedFetch, settings } = useAuth()
    const { jobs, refreshJobDetail, reportJob } = useJobs()
    const navigate = useNavigate()
    const fileInputRef = React.useRef<HTMLInputElement | null>(null)
    const itemsRef = React.useRef<DemoItem[]>([])
    const runningIdsRef = React.useRef<Set<string>>(new Set())
    const localIdRef = React.useRef(0)

    const [items, setItems] = React.useState<DemoItem[]>([])
    const [selectedPlayback, setSelectedPlayback] = React.useState<{ job: JobDetail; file: File } | null>(null)
    const [isSampleDownloading, setIsSampleDownloading] = React.useState(false)
    const [openingJobId, setOpeningJobId] = React.useState<string | null>(null)

    React.useEffect(() => {
        itemsRef.current = items
    }, [items])

    const jobsById = React.useMemo(() => new Map(jobs.map(j => [j.id, j] as const)), [jobs])
    const jobsBySha = React.useMemo(() => new Map(jobs.map(j => [j.sha256, j] as const)), [jobs])

    const hasLocalBusy = items.some(i => i.phase === 'queued' || i.phase === 'hashing' || i.phase === 'uploading')
    const hasActiveOrUnknownJob = items.some(i => {
        if (i.phase === 'error') return false
        if (!i.jobId) return i.phase === 'waiting'
        const job = jobsById.get(i.jobId)
        return !job || !isTerminal(job.status)
    })
    const shouldWarnOnLeave = hasLocalBusy || hasActiveOrUnknownJob

    React.useEffect(() => {
        if (!shouldWarnOnLeave) return
        const warning = t('screens.demo.leaveWarning')
        const onBeforeUnload = (event: BeforeUnloadEvent) => {
            event.preventDefault()
            event.returnValue = warning
            return warning
        }
        window.addEventListener('beforeunload', onBeforeUnload)
        return () => window.removeEventListener('beforeunload', onBeforeUnload)
    }, [shouldWarnOnLeave, t])

    const makeDemoItem = React.useCallback((file: File): DemoItem => {
        localIdRef.current += 1
        return {
            id: `demo-${Date.now()}-${localIdRef.current}`,
            createdAtMs: Date.now(),
            file,
            sha256: null,
            jobId: null,
            progressPct: null,
            phase: 'queued',
            error: null,
            duplicateSkipped: false,
        }
    }, [])

    const patchItem = React.useCallback((id: string, patch: Partial<DemoItem>) => {
        setItems(prev => prev.map(item => (item.id === id ? { ...item, ...patch } : item)))
    }, [])

    const enqueueFiles = React.useCallback(
        (files: File[]) => {
            if (files.length === 0) return
            setItems(prev => [...files.map(makeDemoItem), ...prev])
        },
        [makeDemoItem]
    )

    const processItem = React.useCallback(
        async (itemId: string) => {
            if (runningIdsRef.current.has(itemId)) return
            const item = itemsRef.current.find(i => i.id === itemId)
            if (!item || item.phase !== 'queued') return

            runningIdsRef.current.add(itemId)
            patchItem(itemId, { phase: 'hashing', progressPct: null, error: null, duplicateSkipped: false })

            try {
                const sha256 = await computeSha256(item.file)
                patchItem(itemId, { sha256 })

                const result = await uploadVideoFile({
                    file: item.file,
                    quality: settings.uploadQuality,
                    authorizedFetch,
                    onProgress: p => patchItem(itemId, { progressPct: Math.round(p * 100) }),
                    onStarted: () => patchItem(itemId, { phase: 'uploading' }),
                    sha256,
                })

                if (result === 'skipped') {
                    patchItem(itemId, {
                        phase: 'waiting',
                        jobId: null,
                        progressPct: null,
                        duplicateSkipped: true,
                    })
                    return
                }

                patchItem(itemId, {
                    jobId: result,
                    phase: 'waiting',
                    progressPct: null,
                    duplicateSkipped: false,
                })
            } catch (e: any) {
                patchItem(itemId, {
                    phase: 'error',
                    error: e?.message || String(e),
                    progressPct: null,
                })
            } finally {
                runningIdsRef.current.delete(itemId)
                setItems(prev => [...prev])
            }
        },
        [authorizedFetch, patchItem, settings.uploadQuality]
    )

    React.useEffect(() => {
        const available = DEMO_CONCURRENT_WORKERS - runningIdsRef.current.size
        if (available <= 0) return
        const nextQueuedIds = items.filter(i => i.phase === 'queued').slice(0, available).map(i => i.id)
        for (const id of nextQueuedIds) void processItem(id)
    }, [items, processItem])

    const startSampleUpload = React.useCallback(async () => {
        if (isSampleDownloading) return
        try {
            setIsSampleDownloading(true)
            const res = await fetch('/sample_video.mp4')
            if (!res.ok) throw new Error(await res.text())
            const blob = await res.blob()
            const file = new File([blob], 'GybeLock-Demo.mp4', { type: 'video/mp4' })
            enqueueFiles([file])
        } catch (e: any) {
            const file = new File([], `Sample-download-failed-${Date.now()}.mp4`, { type: 'video/mp4' })
            setItems(prev => [
                {
                    ...makeDemoItem(file),
                    phase: 'error',
                    error: e?.message || String(e),
                    createdAtMs: Date.now(),
                },
                ...prev,
            ])
        } finally {
            setIsSampleDownloading(false)
        }
    }, [enqueueFiles, isSampleDownloading, makeDemoItem])

    React.useEffect(() => {
        setItems(prev => {
            let changed = false
            const next = prev.map(item => {
                if (item.jobId || !item.sha256) return item
                const match = jobsBySha.get(item.sha256)
                if (!match) return item
                changed = true
                return { ...item, jobId: match.id }
            })
            return changed ? next : prev
        })
    }, [jobsBySha])

    const handlePick = React.useCallback(() => {
        fileInputRef.current?.click()
    }, [])

    const handleGoHome = React.useCallback(() => {
        if (!shouldWarnOnLeave) {
            onGoHome()
            return
        }
        if (!window.confirm(t('screens.demo.leaveWarning'))) return
        onGoHome()
    }, [onGoHome, shouldWarnOnLeave, t])

    const openPlayerForItem = React.useCallback(
        async (itemId: string) => {
            const item = itemsRef.current.find(i => i.id === itemId)
            if (!item) return
            const job =
                (item.jobId ? jobsById.get(item.jobId) : null) ??
                (item.sha256 ? jobsBySha.get(item.sha256) : null) ??
                null
            if (!job || job.status !== 'succeeded') return
            setOpeningJobId(job.id)
            try {
                const detail = await refreshJobDetail(job.id)
                setSelectedPlayback({ job: detail, file: item.file })
            } finally {
                setOpeningJobId(null)
            }
        },
        [jobsById, jobsBySha, refreshJobDetail]
    )

    const goToAnalyzer = React.useCallback(async () => {
        navigate('/analyzer')
    }, [navigate])

    const sortedItems = React.useMemo(() => {
        const rank = (item: DemoItem): number => {
            if (item.phase === 'error') return 2
            if (!item.jobId) {
                return item.phase === 'waiting' ? 0 : 0
            }
            const job = jobsById.get(item.jobId)
            if (!job) return 0
            if (job.status === 'succeeded') return 1
            if (job.status === 'failed' || job.status === 'canceled') return 2
            return 0
        }

        return [...items].sort((a, b) => {
            const ra = rank(a)
            const rb = rank(b)
            if (ra !== rb) return ra - rb
            return b.createdAtMs - a.createdAtMs
        })
    }, [items, jobsById])

    const makePlaceholderJob = React.useCallback(
        (item: DemoItem): JobSummary => ({
            id: item.jobId ?? `local:${item.id}`,
            status:
                item.phase === 'error'
                    ? 'failed'
                    : item.phase === 'hashing'
                        ? 'starting'
                        : item.phase === 'waiting'
                            ? 'starting'
                            : 'uploading',
            created_at: new Date(item.createdAtMs).toISOString(),
            updated_at: new Date().toISOString(),
            sha256: item.sha256 ?? `local:${item.id}`,
            dominant_orientation: 0,
        }),
        []
    )

    return (
        <div className="min-h-dvh bg-white text-slate-900">
            <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                <div className="mx-auto max-w-3xl px-4 sm:px-6 py-3 flex items-center gap-3">
                    <LogoButton onClick={handleGoHome} />
                    <div className="flex items-center gap-2 min-w-0">
                        <div className="text-sm font-semibold text-slate-900 truncate">{t('screens.demo.hero.title')}</div>
                        <span className="text-[11px] font-semibold text-brand-700 bg-brand-50 border border-brand-600/20 px-2 py-0.5 rounded-full">
                            {t('screens.demo.title')}
                        </span>
                    </div>
                    <div className="flex-1" />
                    <div className="flex flex-col items-end gap-1">
                        <Button
                            variant="brandOutline"
                            size="sm"
                            onClick={() => void goToAnalyzer()}
                            text={t('screens.demo.upgrade.cta')}
                        />
                    </div>
                    <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        accept="video/mp4"
                        className="hidden"
                        onChange={e => {
                            const picked = Array.from(e.target.files ?? [])
                            e.currentTarget.value = ''
                            enqueueFiles(picked)
                        }}
                    />
                </div>
            </header>

            <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-6">
                <div className="mx-auto max-w-3xl flex flex-col gap-6">
                    {/* Video block */}
                    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                        <div>
                            <div className="text-sm text-slate-600">{t('screens.demo.hero.body')}</div>
                        </div>

                        <div className="mt-5">
                            <div className="flex flex-wrap items-center gap-3">
                                <Button
                                    variant="primary"
                                    size="md"
                                    onClick={handlePick}
                                    text={t('screens.demo.actions.selectVideo')}
                                />
                                <div className="text-xs text-slate-500">{t('screens.demo.processingEta')}</div>
                                <div className="flex-1" />
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => void startSampleUpload()}
                                    disabled={isSampleDownloading}
                                    isPending={isSampleDownloading}
                                    text={t('screens.demo.actions.sampleVideo')}
                                />
                            </div>
                        </div>
                    </section>

                    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                        <div className="flex flex-wrap items-center gap-3">
                            <div>
                                <div className="text-sm font-semibold text-slate-900">{t('screens.demo.grid.title')}</div>
                                <div className="text-xs text-slate-500">
                                    {items.length === 0
                                        ? t('screens.demo.grid.empty')
                                        : t('screens.demo.grid.processingHint')}
                                </div>
                            </div>
                            <div className="flex-1" />
                        </div>

                        {items.length === 0 ? (
                            <div className="mt-5 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-6">
                                <div className="text-sm text-slate-600">{t('screens.demo.grid.empty')}</div>
                            </div>
                        ) : (
                            <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                                {sortedItems.map(item => {
                                    const currentJob =
                                        (item.jobId ? jobsById.get(item.jobId) ?? null : null) ??
                                        (item.sha256 ? jobsBySha.get(item.sha256) ?? null : null)
                                    const thumbnailJob = currentJob ?? makePlaceholderJob(item)
                                    const canOpen = !!currentJob && currentJob.status === 'succeeded'
                                    const isOpening = !!currentJob && openingJobId === currentJob.id
                                    const showError = !!item.error && !item.duplicateSkipped

                                    return (
                                        <div key={item.id} className="rounded-xl border border-slate-200 p-3">
                                            <div
                                                className={`relative ${canOpen ? 'cursor-pointer' : 'cursor-default'} ${isOpening ? 'opacity-70' : ''
                                                    }`}
                                                onClick={() => {
                                                    if (!canOpen) return
                                                    void openPlayerForItem(item.id)
                                                }}
                                                role={canOpen ? 'button' : undefined}
                                                tabIndex={canOpen ? 0 : -1}
                                                onKeyDown={e => {
                                                    if (!canOpen) return
                                                    if (e.key === 'Enter' || e.key === ' ') void openPlayerForItem(item.id)
                                                }}
                                            >
                                                <JobThumbnail
                                                    job={thumbnailJob}
                                                    videoSource={{ kind: 'file', file: item.file }}
                                                    playable={canOpen}
                                                />
                                                {isOpening && (
                                                    <div className="absolute inset-0 flex items-center justify-center">
                                                        <div className="rounded bg-black/60 px-2 py-1 text-xs text-white">
                                                            {t('components.jobList.opening')}
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                            {item.phase === 'uploading' && (
                                                <div className="mt-2">
                                                    <UploadProgressBar percent={item.progressPct ?? 0} />
                                                </div>
                                            )}
                                            {showError && (
                                                <div className="mt-2 text-xs text-red-700 break-words">{item.error}</div>
                                            )}
                                        </div>
                                    )
                                })}
                            </div>
                        )}
                    </section>

                    {/* Upgrade block */}
                    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                        <div className="text-sm font-semibold text-slate-900">{t('screens.demo.upgrade.title')}</div>
                        <ul className="mt-2 list-disc pl-5 space-y-1 text-sm text-slate-700">
                            <li>{t('screens.demo.upgrade.bullets.folder')}</li>
                            <li>{t('screens.demo.upgrade.bullets.workflow')}</li>
                            <li>{t('screens.demo.upgrade.bullets.parallel')}</li>
                        </ul>
                        <div className="mt-4">
                            <Button
                                variant="brandOutline"
                                size="md"
                                onClick={() => void goToAnalyzer()}
                                text={t('screens.demo.upgrade.cta')}
                            />
                        </div>
                    </section>
                </div>

                {selectedPlayback && (
                    <PlayerModal
                        job={selectedPlayback.job}
                        videoSource={{ kind: 'file', file: selectedPlayback.file }}
                        onClose={() => setSelectedPlayback(null)}
                        onReport={reportJob}
                        showDemoTutorialTips
                    />
                )}
            </main>
        </div>
    )
}
