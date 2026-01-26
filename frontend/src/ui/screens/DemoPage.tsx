import React from 'react'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../auth/AuthProvider'
import { Button } from '../components/Button'
import JobThumbnail from '../components/JobThumbnail'
import { LogoButton } from '../components/LogoButton'
import { PlayerModal } from '../components/PlayerModal'
import { Spinner } from '../components/Spinner'
import { useJobs } from '../hooks/useJobs'
import { computeSha256 } from '../utils/localFileIndex'
import { uploadVideoFile } from '../utils/uploader'
import type { JobDetail, JobSummary } from '../types'
import { useNavigate } from 'react-router-dom'
import { auth } from '../../firebase'
import { signOut } from 'firebase/auth'

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

export const DemoPage: React.FC<{ onGoHome: () => void }> = ({ onGoHome }) => {
    const { t } = useTranslation()
    const { authorizedFetch, settings } = useAuth()
    const { jobs, refreshJobDetail, reportJob } = useJobs()
    const navigate = useNavigate()
    const fileInputRef = React.useRef<HTMLInputElement | null>(null)

    const [selectedFile, setSelectedFile] = React.useState<File | null>(null)
    const [selectedFileSha256, setSelectedFileSha256] = React.useState<string | null>(null)
    const [jobId, setJobId] = React.useState<string | null>(null)
    const [progressPct, setProgressPct] = React.useState<number | null>(null)
    const [phase, setPhase] = React.useState<'idle' | 'hashing' | 'uploading' | 'waiting'>('idle')
    const [error, setError] = React.useState<string | null>(null)
    const [selectedJob, setSelectedJob] = React.useState<JobDetail | null>(null)
    const [isSampleDownloading, setIsSampleDownloading] = React.useState(false)
    const autoOpenedRef = React.useRef(false)

    const currentJob = React.useMemo(() => (jobId ? jobs.find(j => j.id === jobId) ?? null : null), [jobs, jobId])

    const busy = phase === 'hashing' || phase === 'uploading'
    const shouldWarnOnLeave = busy || (!!jobId && (!currentJob || !isTerminal(currentJob.status)))

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

    const reset = React.useCallback(() => {
        setSelectedFile(null)
        setSelectedFileSha256(null)
        setJobId(null)
        setProgressPct(null)
        setPhase('idle')
        setError(null)
        setSelectedJob(null)
        autoOpenedRef.current = false
    }, [])

    const startUpload = React.useCallback(
        async (file: File) => {
            reset()
            setSelectedFile(file)
            setError(null)

            try {
                setPhase('hashing')
                setProgressPct(null)
                const sha256 = await computeSha256(file)
                setSelectedFileSha256(sha256)

                setPhase('uploading')
                const result = await uploadVideoFile({
                    file,
                    quality: settings.uploadQuality,
                    authorizedFetch,
                    onProgress: p => setProgressPct(Math.round(p * 100)),
                    onStarted: () => setPhase('uploading'),
                    sha256,
                })

                if (result === 'skipped') {
                    // Duplicate/already-processed: job is associated to the user, so we can resolve it
                    // from realtime jobs by sha256 and still open the player.
                    setPhase('waiting')
                    setJobId(null)
                    setProgressPct(null)
                    return
                }

                setJobId(result)
                setPhase('waiting')
            } catch (e: any) {
                setError(e?.message || String(e))
                setPhase('idle')
            }
        },
        [authorizedFetch, reset, settings.uploadQuality, t]
    )

    const startSampleUpload = React.useCallback(async () => {
        if (busy || isSampleDownloading) return
        try {
            setIsSampleDownloading(true)
            setError(null)
            const res = await fetch('/sample_video.mp4')
            if (!res.ok) throw new Error(await res.text())
            const blob = await res.blob()
            const file = new File([blob], 'GybeLock-Demo.mp4', { type: 'video/mp4' })
            await startUpload(file)
        } catch (e: any) {
            setError(e?.message || String(e))
        } finally {
            setIsSampleDownloading(false)
        }
    }, [busy, isSampleDownloading, startUpload])

    React.useEffect(() => {
        if (!selectedFileSha256) return
        if (jobId) return
        const match = jobs.find(j => j.sha256 === selectedFileSha256)
        if (!match) return
        setJobId(match.id)
    }, [jobId, jobs, selectedFileSha256])

    React.useEffect(() => {
        if (!jobId) return
        if (!selectedFile) return
        if (!currentJob) return
        if (currentJob.status !== 'succeeded') return
        if (autoOpenedRef.current) return

        autoOpenedRef.current = true
        void (async () => {
            try {
                const detail = await refreshJobDetail(jobId)
                setSelectedJob(detail)
            } catch (e: any) {
                setError(e?.message || String(e))
            }
        })()
    }, [currentJob, jobId, refreshJobDetail, selectedFile])

    const handlePick = React.useCallback(() => {
        if (busy) return
        fileInputRef.current?.click()
    }, [busy])

    const handleGoHome = React.useCallback(() => {
        if (!shouldWarnOnLeave) {
            onGoHome()
            return
        }
        if (!window.confirm(t('screens.demo.leaveWarning'))) return
        onGoHome()
    }, [onGoHome, shouldWarnOnLeave, t])

    const openPlayerForCurrentJob = React.useCallback(async () => {
        if (!jobId) return
        if (!currentJob || currentJob.status !== 'succeeded') return
        const detail = await refreshJobDetail(jobId)
        setSelectedJob(detail)
    }, [currentJob, jobId, refreshJobDetail])

    const goToAnalyzer = React.useCallback(async () => {
        // Demo uses anonymous/ephemeral auth; sign out so the full Analyzer starts at login/signup.
        try {
            await signOut(auth)
        } catch { }
        navigate('/analyzer')
    }, [navigate])

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
                        <div className="hidden sm:block text-[10px] leading-4 text-slate-500 text-right">
                            {t('common.fullAnalyzerBetaFreeNote')}
                        </div>
                    </div>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="video/mp4"
                        className="hidden"
                        onChange={e => {
                            const f = e.target.files?.[0] ?? null
                            e.currentTarget.value = ''
                            if (!f) return
                            void startUpload(f)
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
                            {selectedFile && jobId && currentJob ? (
                                <div>
                                    <div
                                        className={`${currentJob.status === 'succeeded' ? 'cursor-pointer' : 'cursor-default'
                                            }`}
                                        onClick={() => {
                                            if (currentJob.status !== 'succeeded') return
                                            void openPlayerForCurrentJob()
                                        }}
                                        role={currentJob.status === 'succeeded' ? 'button' : undefined}
                                        tabIndex={currentJob.status === 'succeeded' ? 0 : -1}
                                        onKeyDown={e => {
                                            if (currentJob.status !== 'succeeded') return
                                            if (e.key === 'Enter' || e.key === ' ') void openPlayerForCurrentJob()
                                        }}
                                    >
                                        <JobThumbnail
                                            job={currentJob}
                                            videoSource={{ kind: 'file', file: selectedFile }}
                                            playable={currentJob.status === 'succeeded'}
                                            layout="wide"
                                        />
                                    </div>
                                    {currentJob.status === 'uploading' && <div className="mt-2"><UploadProgressBar percent={progressPct ?? 0} /></div>}
                                    <div className="mt-2 text-sm font-semibold text-slate-900 break-words">
                                        {selectedFile.name}
                                    </div>
                                    {error && <div className="mt-2 text-sm text-red-700 break-words">{error}</div>}
                                    {jobId && currentJob && currentJob.status === 'failed' && (
                                        <div className="mt-2 text-sm text-red-700">{t('screens.demo.failed')}</div>
                                    )}
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    <div className="flex items-center gap-3 text-sm text-slate-600">
                                        {(phase === 'hashing' || phase === 'uploading') && <Spinner size="small" />}
                                        {phase === 'hashing'
                                            ? t('screens.demo.status.hashing')
                                            : phase === 'uploading'
                                                ? `${t('screens.demo.status.uploading')}${
                                                      progressPct != null ? ` (${progressPct}%)` : ''
                                                  }`
                                                : t('screens.demo.status.waiting')}
                                    </div>

                                    {phase === 'uploading' && <UploadProgressBar percent={progressPct ?? 0} />}
                                </div>
                            )}
                        </div>

                        <div className="mt-5">
                            <div className="flex flex-wrap items-center gap-3">
                                <Button
                                    variant="primary"
                                    size="md"
                                    onClick={handlePick}
                                    disabled={busy}
                                    text={t('screens.demo.actions.selectVideo')}
                                />
                                <div className="text-xs text-slate-500">{t('screens.demo.processingEta')}</div>
                                <div className="flex-1" />
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => void startSampleUpload()}
                                    disabled={busy || isSampleDownloading}
                                    isPending={isSampleDownloading}
                                    text={t('screens.demo.actions.sampleVideo')}
                                />
                            </div>
                        </div>
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
                            <div className="mt-2 text-xs text-slate-500">{t('common.fullAnalyzerBetaFreeNote')}</div>
                        </div>
                    </section>
                </div>

                {selectedJob && selectedFile && (
                    <PlayerModal
                        job={selectedJob}
                        videoSource={{ kind: 'file', file: selectedFile }}
                        onClose={() => setSelectedJob(null)}
                        onReport={reportJob}
                        showDemoTutorialTips
                    />
                )}
            </main>
        </div>
    )
}
