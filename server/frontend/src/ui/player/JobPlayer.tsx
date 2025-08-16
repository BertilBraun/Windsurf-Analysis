import React from 'react'
import { JobDetail, ReportType } from '../types'
import { useAuth } from '../auth/AuthProvider'

type VideoProperties = {
    fps: number
    width: number
    height: number
    total_frames: number
}

type Detection = {
    frame: number
    track_id: number
    bbox: [number, number, number, number] // x, y, w, h in pixels
}

type Track = {
    id: number
    detections: Detection[]
}

type Metadata = {
    input_video_path: string
    video_properties: VideoProperties
    tracks: Track[]
}

function getBasename(p: string): string {
    const parts = p.split(/[/\\]/)
    return parts[parts.length - 1]
}

function buildDetectionsByFrame(tracks: Track[], totalFrames: number): Map<number, Detection[]> {
    const map = new Map<number, Detection[]>()
    for (let i = 0; i < totalFrames; i++) map.set(i, [])
    for (const t of tracks) {
        for (const d of t.detections) {
            const arr = map.get(d.frame)
            if (arr) arr.push(d)
            else map.set(d.frame, [d])
        }
    }
    return map
}

function clamp(x: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, x))
}

export const JobPlayer: React.FC<{ job: JobDetail; onClose: () => void; onDeleted?: () => void }> = ({
    job,
    onClose,
    onDeleted,
}) => {
    const { authorizedFetch } = useAuth()
    const [metadata, setMetadata] = React.useState<Metadata | null>(null)
    const [videoUrl, setVideoUrl] = React.useState<string | null>(null)
    const [hudMessage, setHudMessage] = React.useState<string | null>(null)

    const videoRef = React.useRef<HTMLVideoElement | null>(null)
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null)

    const [currentFrame, setCurrentFrame] = React.useState(0)
    const [isPlaying, setIsPlaying] = React.useState(false)
    const [playbackSpeed, setPlaybackSpeed] = React.useState(1.0)
    const [currentMode, setCurrentMode] = React.useState<'overview' | 'detailed'>('overview')
    const [currentTrackId, setCurrentTrackId] = React.useState<number | null>(null)

    const timerRef = React.useRef<number | null>(null)
    const detectionsByFrameRef = React.useRef<Map<number, Detection[]>>(new Map())

    const [isReportOpen, setIsReportOpen] = React.useState(false)
    const [reportType, setReportType] = React.useState<ReportType>('other')
    const [reportMessage, setReportMessage] = React.useState('')
    const [isSubmittingReport, setIsSubmittingReport] = React.useState(false)
    const [isDeleting, setIsDeleting] = React.useState(false)
    const [isVideoReady, setIsVideoReady] = React.useState(false)

    React.useEffect(() => {
        const raw = (job.results_json || {}) as any
        if (!raw || !raw.video_properties || !raw.tracks) return
        const meta: Metadata = {
            input_video_path: String(job.original_file_path || ''),
            video_properties: {
                fps: Number(raw.video_properties.fps || 30),
                width: Number(raw.video_properties.width || 0),
                height: Number(raw.video_properties.height || 0),
                total_frames: Number(raw.video_properties.total_frames || 0),
            },
            tracks: (raw.tracks as any[]).map((t, idx) => ({
                id: Number(t.track_id ?? idx),
                detections: (t.detections || []).map((d: any) => {
                    const x1 = Number(d.bbox[0])
                    const y1 = Number(d.bbox[1])
                    const x2 = Number(d.bbox[2])
                    const y2 = Number(d.bbox[3])
                    return {
                        frame: Number(d.frame_idx),
                        track_id: Number(t.track_id ?? idx),
                        bbox: [x1, y1, x2 - x1, y2 - y1] as [number, number, number, number],
                    }
                }),
            })),
        }
        setMetadata(meta)
        detectionsByFrameRef.current = buildDetectionsByFrame(meta.tracks, meta.video_properties.total_frames)
        setCurrentFrame(0)
        setIsPlaying(false)
        setCurrentMode('overview')
        setCurrentTrackId(null)
    }, [job])

    const pickVideo = async () => {
        const input = document.createElement('input')
        input.type = 'file'
        input.accept = 'video/*'
        input.onchange = () => {
            if (!input.files || input.files.length === 0) return
            const file = input.files[0]
            if (metadata?.input_video_path) {
                const expected = getBasename(metadata.input_video_path)
                if (getBasename(file.name) !== expected) {
                    alert(`Selected file "${file.name}" does not match expected "${expected}"`)
                }
            }
            const url = URL.createObjectURL(file)
            setIsVideoReady(false)
            setVideoUrl(url)
        }
        input.click()
    }

    const showHud = (msg: string) => {
        setHudMessage(msg)
        window.setTimeout(() => setHudMessage(null), 1000)
    }

    const submitReport = async () => {
        if (!reportMessage) return
        setIsSubmittingReport(true)
        try {
            await authorizedFetch(`/jobs/${job.id}/report`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: reportType, message: reportMessage }),
            })
            setIsReportOpen(false)
            setReportMessage('')
            setReportType('other')
            showHud('Report submitted')
        } finally {
            setIsSubmittingReport(false)
        }
    }

    const deleteJob = async () => {
        if (!window.confirm('Delete this job?')) return
        setIsDeleting(true)
        try {
            await authorizedFetch(`/jobs/${job.id}`, { method: 'DELETE' })
            if (onDeleted) onDeleted()
            onClose()
        } finally {
            setIsDeleting(false)
        }
    }

    const updateTimerInterval = React.useCallback(() => {
        if (!metadata) return
        if (timerRef.current) window.clearInterval(timerRef.current)
        const fps = Math.max(0.1, metadata.video_properties.fps * playbackSpeed)
        const intervalMs = Math.max(5, Math.min(1000.0, 1000.0 / fps))
        timerRef.current = window.setInterval(() => {
            setCurrentFrame(prev => {
                const next = Math.min(prev + 1, (metadata?.video_properties.total_frames || 1) - 1)
                return next
            })
        }, intervalMs)
    }, [metadata, playbackSpeed])

    React.useEffect(() => {
        if (isPlaying) updateTimerInterval()
        return () => {
            if (timerRef.current) window.clearInterval(timerRef.current)
        }
    }, [isPlaying, updateTimerInterval])

    React.useEffect(() => {
        const video = videoRef.current
        if (!video) return
        const onLoadedMeta = () => setIsVideoReady(true)
        const onLoadedData = () => setIsVideoReady(true)
        const onCanPlay = () => setIsVideoReady(true)
        video.addEventListener('loadedmetadata', onLoadedMeta)
        video.addEventListener('loadeddata', onLoadedData)
        video.addEventListener('canplay', onCanPlay)
        // force load in case the browser delays
        try {
            video.load()
        } catch {}
        if (video.readyState >= 1) setIsVideoReady(true)
        return () => {
            video.removeEventListener('loadedmetadata', onLoadedMeta)
            video.removeEventListener('loadeddata', onLoadedData)
            video.removeEventListener('canplay', onCanPlay)
        }
    }, [videoUrl])

    React.useEffect(() => {
        const video = videoRef.current
        const canvas = canvasRef.current
        if (!video || !canvas || !metadata || !videoUrl) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Use video intrinsic dimensions to match pixel space; fallback to metadata or sensible defaults
        const cw = video.videoWidth || metadata.video_properties.width || 640
        const ch = video.videoHeight || metadata.video_properties.height || 360
        canvas.width = cw
        canvas.height = ch

        // Draw routine
        const draw = () => {
            try {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
            } catch {}
            const dets = detectionsByFrameRef.current.get(currentFrame) || []
            for (const d of dets) {
                const [x, y, w, h] = d.bbox
                ctx.strokeStyle = '#22c55e'
                ctx.lineWidth = 2
                ctx.strokeRect(x, y, w, h)
                ctx.fillStyle = 'rgba(34,197,94,0.2)'
                ctx.fillRect(x, y, w, h)
                ctx.fillStyle = 'white'
                ctx.font = '12px monospace'
                ctx.fillText(String(d.track_id), x + 4, y + 14)
            }
            if (hudMessage) {
                ctx.fillStyle = 'rgba(0,0,0,0.6)'
                ctx.fillRect(8, 8, 160, 22)
                ctx.fillStyle = 'white'
                ctx.font = '14px sans-serif'
                ctx.fillText(hudMessage, 14, 24)
            }
        }

        // Keep frame/time in sync and draw on relevant events
        const syncAndDraw = () => {
            const t = Math.max(0, currentFrame) / Math.max(1, metadata.video_properties.fps)
            // Only adjust currentTime if drift is significant to avoid thrashing
            if (Math.abs(video.currentTime - t) > 1e-3) {
                video.currentTime = t
            }
            requestAnimationFrame(draw)
        }

        const onSeeked = () => draw()
        const onTimeUpdate = () => draw()
        const onLoadedData = () => draw()
        const onCanPlay = () => draw()
        video.addEventListener('seeked', onSeeked)
        video.addEventListener('timeupdate', onTimeUpdate)
        video.addEventListener('loadeddata', onLoadedData)
        video.addEventListener('canplay', onCanPlay)

        syncAndDraw()

        return () => {
            video.removeEventListener('seeked', onSeeked)
            video.removeEventListener('timeupdate', onTimeUpdate)
            video.removeEventListener('loadeddata', onLoadedData)
            video.removeEventListener('canplay', onCanPlay)
        }
    }, [currentFrame, metadata, hudMessage, videoUrl])

    React.useEffect(() => {
        if (!metadata) return
        if (currentMode === 'detailed' && currentTrackId != null) {
            const dets = detectionsByFrameRef.current.get(currentFrame) || []
            const has = dets.some(d => d.track_id === currentTrackId)
            if (!has) {
                setCurrentMode('overview')
                setCurrentTrackId(null)
            }
        }
    }, [currentFrame, currentMode, currentTrackId, metadata])

    React.useEffect(
        () => () => {
            if (timerRef.current) window.clearInterval(timerRef.current)
            if (videoUrl) URL.revokeObjectURL(videoUrl)
        },
        []
    )

    const bumpSpeed = (down: boolean) => {
        const rates = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        const idx = rates.indexOf(playbackSpeed)
        const next = rates[clamp(idx == -1 ? 2 : down ? idx - 1 : idx + 1, 0, rates.length - 1)]
        setPlaybackSpeed(next)
        showHud(`Speed: ${next}x`)
    }

    const seekToFrame = (frame: number) => {
        if (!metadata) return
        const clamped = Math.max(0, Math.min(frame, metadata.video_properties.total_frames - 1))
        setCurrentFrame(clamped)
    }

    const handleKeyDown = (event: KeyboardEvent) => {
        if (!metadata) return
        const active = document.activeElement as HTMLElement | null
        if (
            active &&
            (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || (active as any).isContentEditable)
        ) {
            return
        }
        if (event.key === ' ') {
            event.preventDefault()
            setIsPlaying(p => !p)
            return
        }
        if (event.key === 'ArrowLeft' && !event.ctrlKey && !event.shiftKey) {
            event.preventDefault()
            setIsPlaying(false)
            seekToFrame(currentFrame - 1)
            return
        }
        if (event.key === 'ArrowRight' && !event.ctrlKey && !event.shiftKey) {
            event.preventDefault()
            setIsPlaying(false)
            seekToFrame(currentFrame + 1)
            return
        }
        if (event.ctrlKey && event.key === 'ArrowLeft') {
            event.preventDefault()
            const delta = Math.round(metadata.video_properties.fps * 30)
            seekToFrame(currentFrame - delta)
            return
        }
        if (event.ctrlKey && event.key === 'ArrowRight') {
            event.preventDefault()
            const delta = Math.round(metadata.video_properties.fps * 30)
            seekToFrame(currentFrame + delta)
            return
        }
        if (event.shiftKey && event.key === 'ArrowLeft') {
            event.preventDefault()
            const delta = Math.round(metadata.video_properties.fps * 5)
            seekToFrame(currentFrame - delta)
            return
        }
        if (event.shiftKey && event.key === 'ArrowRight') {
            event.preventDefault()
            const delta = Math.round(metadata.video_properties.fps * 5)
            seekToFrame(currentFrame + delta)
            return
        }
        if (event.key.toLowerCase() === 'escape') {
            setCurrentMode('overview')
            setCurrentTrackId(null)
            return
        }
        if (event.key === '-' || event.key === '_') {
            bumpSpeed(true)
            return
        }
        if (event.key === '+' || event.key === '=') {
            bumpSpeed(false)
            return
        }
    }

    React.useEffect(() => {
        const onKey = (e: KeyboardEvent) => handleKeyDown(e)
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    })

    const onCanvasClick: React.MouseEventHandler<HTMLCanvasElement> = e => {
        if (!metadata) return
        const canvas = canvasRef.current
        if (!canvas) return
        const rect = canvas.getBoundingClientRect()
        const cssX = e.clientX - rect.left
        const cssY = e.clientY - rect.top
        const scaleX = canvas.width / rect.width
        const scaleY = canvas.height / rect.height
        const x = cssX * scaleX
        const y = cssY * scaleY
        const dets = detectionsByFrameRef.current.get(currentFrame) || []
        for (const d of dets) {
            const [bx, by, bw, bh] = d.bbox
            if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
                setCurrentMode('detailed')
                setCurrentTrackId(d.track_id)
                return
            }
        }
    }

    if (!metadata) {
        return (
            <div>
                <h4>Results not available</h4>
                <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(job.results_json ?? {}, null, 2)}</pre>
                <button onClick={onClose}>Close</button>
            </div>
        )
    }

    return (
        <>
            <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <div>
                            <strong>Job</strong> {getBasename(job.original_file_path) || 'unknown.mp4'}
                        </div>
                        <div style={{ fontSize: 12, color: '#6b7280' }}>
                            File: {getBasename(job.original_file_path)} Job ID: {job.id}
                        </div>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <button onClick={onClose}>Close</button>
                        <button onClick={pickVideo}>{videoUrl ? 'Change video' : 'Select video'}</button>
                        <button onClick={() => setIsReportOpen(true)}>Report</button>
                        <button onClick={deleteJob} disabled={isDeleting} style={{ color: '#ef4444' }}>
                            {isDeleting ? 'Deleting…' : 'Delete'}
                        </button>
                    </div>
                </div>

                <div style={{ marginTop: 8, border: '1px solid #eee', borderRadius: 8, padding: 8 }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {!videoUrl && (
                            <div style={{ padding: 24, textAlign: 'center', color: '#6b7280' }}>
                                <div style={{ marginBottom: 8 }}>Select the original video file to start playback.</div>
                                <button onClick={pickVideo}>Select video</button>
                            </div>
                        )}
                        {videoUrl && (
                            <>
                                <canvas
                                    ref={canvasRef}
                                    onClick={onCanvasClick}
                                    style={{ width: '100%', maxHeight: 640, background: '#000' }}
                                />
                                <video
                                    ref={videoRef}
                                    src={videoUrl || undefined}
                                    style={{ display: 'none' }}
                                    preload="metadata"
                                    playsInline
                                    muted
                                    onLoadedMetadata={() => setIsVideoReady(true)}
                                    onLoadedData={() => setIsVideoReady(true)}
                                    onCanPlay={() => setIsVideoReady(true)}
                                />
                            </>
                        )}

                        <input
                            type="range"
                            min={0}
                            max={Math.max(0, metadata.video_properties.total_frames - 1)}
                            value={currentFrame}
                            onChange={e => seekToFrame(Number(e.target.value))}
                        />

                        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                            <button onClick={() => setIsPlaying(p => !p)} disabled={!videoUrl}>
                                {isPlaying ? 'Pause' : 'Play'}
                            </button>
                            <button onClick={() => bumpSpeed(true)} disabled={!videoUrl}>
                                Speed -
                            </button>
                            <button onClick={() => bumpSpeed(false)} disabled={!videoUrl}>
                                Speed +
                            </button>
                            <span style={{ fontFamily: 'monospace' }}>Speed: {playbackSpeed}x</span>
                            <span style={{ marginLeft: 12, fontFamily: 'monospace' }}>
                                Frame {currentFrame + 1}/{metadata.video_properties.total_frames}
                            </span>
                            <span style={{ marginLeft: 12 }}>
                                Mode: {currentMode}
                                {currentTrackId != null ? ` (#${currentTrackId})` : ''}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            {isReportOpen && (
                <div
                    style={{
                        position: 'fixed',
                        inset: 0,
                        background: 'rgba(0,0,0,0.5)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                    }}
                >
                    <div
                        style={{
                            background: 'white',
                            padding: 16,
                            borderRadius: 8,
                            width: 420,
                            boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
                        }}
                    >
                        <h4 style={{ marginTop: 0 }}>Report an issue</h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                            <label style={{ fontSize: 12, color: '#374151' }}>Type</label>
                            <select value={reportType} onChange={e => setReportType(e.target.value as ReportType)}>
                                <option value="missed_detection">Missed detection</option>
                                <option value="false_association">False association</option>
                                <option value="other">Other</option>
                            </select>
                            <label style={{ fontSize: 12, color: '#374151' }}>Message</label>
                            <textarea
                                rows={4}
                                value={reportMessage}
                                onChange={e => setReportMessage(e.target.value)}
                                placeholder="Describe the issue..."
                            />
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 12 }}>
                            <button onClick={() => setIsReportOpen(false)} disabled={isSubmittingReport}>
                                {isSubmittingReport ? 'Closing…' : 'Cancel'}
                            </button>
                            <button onClick={submitReport} disabled={!reportMessage || isSubmittingReport}>
                                {isSubmittingReport ? 'Submitting…' : 'Submit'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    )
}
