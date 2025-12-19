import React from 'react'
import { JobDetail, ReportType } from '../types'
import { getFileByRelativePath } from '../utils/fsAccess'
import { getPathsForSha } from '../utils/idb'
import { PlayerState, VideoProperties } from './state'
import { ControlsBar } from './ControlsBar'
import { Timeline } from './Timeline'
import { MAX_SCALE, MIN_SCALE, TARGET_BBOX_HEIGHT_RATIO } from './constants'
import { useZoom } from '../hooks/useZoom'
import { usePlaybackSpeed } from '../hooks/usePlaybackSpeed'
import { useSeeker } from '../hooks/useSeeker'
import { clamp } from '../utils/clamp'
import { drawRotatedToCanvas, getRotatedDimensions } from './rotation'
import { processVideo } from '../../preprocess/preprocess'
import { trackEvent } from '../utils/analytics'

type Props = {
    job: JobDetail
    dirHandle: FileSystemDirectoryHandle | null
    onClose: () => void
    onDelete: (id: string) => void
    onReport: (id: string, type: ReportType, message: string) => void
    onOpenNextJob?: () => void
    onOpenPrevJob?: () => void
}

type OverviewView = {
    zoom: number
    offsetX: number
    offsetY: number
    hoveredTrackId: number | null
}

type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D
type TimedBBox = { time_percent: number; bbox: [number, number, number, number] }

export const CanvasPlayer: React.FC<Props> = ({ job, dirHandle, onClose, onOpenNextJob, onOpenPrevJob }) => {
    const [error, setError] = React.useState<string | null>(null)
    const [exportError, setExportError] = React.useState<string | null>(null)
    const [fileMissing, setFileMissing] = React.useState<boolean>(false)
    const [videoUrl, setVideoUrl] = React.useState<string | null>(null)
    const [sourceFile, setSourceFile] = React.useState<File | null>(null)
    const videoRef = React.useRef<HTMLVideoElement | null>(null)
    const containerRef = React.useRef<HTMLDivElement | null>(null)
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
    const [player, setPlayer] = React.useState<PlayerState | null>(null)
    const { zoom, offset, onWheelZoom } = useZoom(containerRef)
    const { speed, bumpSpeed } = usePlaybackSpeed(1.0)
    const [hoveredTrackId, setHoveredTrackId] = React.useState<number | null>(null)
    const [isExporting, setIsExporting] = React.useState<boolean>(false)
    const [exportProgressPct, setExportProgressPct] = React.useState<number | null>(null)

    // Resolve file URL
    const resolveFileFromRelativePath = React.useCallback(async () => {
        if (!dirHandle) {
            setError('No ingress folder selected')
            return null
        }
        try {
            const candidates: string[] = []
            if (job.local_relative_path) candidates.push(job.local_relative_path)
            if (job.original_checksum_sha256) {
                const extra = await getPathsForSha(String(job.original_checksum_sha256).toLowerCase())
                for (const p of extra) if (!candidates.includes(p)) candidates.push(p)
            }
            if (candidates.length === 0) throw new Error('Missing local mapping for file')

            for (const path of candidates) {
                try {
                    return await getFileByRelativePath(dirHandle, path)
                } catch (e: any) {
                    const msg = String(e?.message || '')
                    const isMissing =
                        e?.name === 'NotFoundError' || /not\s*found|no such file|could not be found/i.test(msg)
                    if (isMissing) continue
                    throw e
                }
            }
            throw new Error('VIDEO FILE NOT FOUND')
        } catch (e: any) {
            const msg = String(e?.message || '')
            const isMissing = e?.name === 'NotFoundError' || /not\s*found|no such file|could not be found/i.test(msg)
            if (isMissing) {
                setFileMissing(true)
                setError('VIDEO FILE NOT FOUND')
            } else {
                setError(msg || 'Failed to access file from folder')
            }
            return null
        }
    }, [dirHandle, job.local_relative_path, job.original_checksum_sha256])

    React.useEffect(() => {
        trackEvent('player_open', { job_id: job.id })
        let revoked: string | null = null
        setVideoUrl(null)
        setError(null)
        setExportError(null)
        setIsExporting(false)
        setExportProgressPct(null)
        setFileMissing(false)
        setSourceFile(null)
        resolveFileFromRelativePath().then(file => {
            if (!file) return
            const url = URL.createObjectURL(file)
            revoked = url
            setVideoUrl(url)
            setSourceFile(file)
            onNewFile(file)
        })
        return () => {
            if (revoked) URL.revokeObjectURL(revoked)
        }
    }, [resolveFileFromRelativePath, job.id])

    // Initialize PlayerState on loadedmetadata
    React.useEffect(() => {
        const v = videoRef.current
        if (!v) return
        const onLoadedMetadata = () => {
            const videoProps: VideoProperties = {
                width: v.videoWidth,
                height: v.videoHeight,
                durationSeconds: v.duration,
            }
            setPlayer(PlayerState.from(job, videoProps))
            // Ensure autoplay starts as soon as metadata is ready
            v.muted = true
            v.play()
        }
        v.addEventListener('loadedmetadata', onLoadedMetadata)
        return () => v.removeEventListener('loadedmetadata', onLoadedMetadata)
    }, [videoUrl, job])

    // Control playback state and speed
    React.useEffect(() => {
        const v = videoRef.current
        if (!v || !player) return
        // During export we temporarily take over playback rate + play/pause.
        if (isExporting) return
        v.defaultPlaybackRate = speed
        v.playbackRate = speed
        if (player.isPlaying) v.play().catch(() => {})
        else v.pause()
    }, [isExporting, player?.isPlaying, speed])

    const togglePlay = React.useCallback(() => setPlayer(p => (p ? p.copy({ isPlaying: !p.isPlaying }) : p)), [])

    // Seeker (frame stepping and seeking)
    const { seekTo, stepNext, stepPrev, onNewFile } = useSeeker(videoRef, player, setPlayer)

    // Helpers for track navigation
    const getSortedTracks = React.useCallback(() => {
        return [...(player?.tracks ?? [])].sort((a, b) => a.start_time_seconds - b.start_time_seconds)
    }, [player?.tracks])

    const goToTrack = React.useCallback(
        (trackId: number, startTimeSec: number) => {
            setPlayer(p => (p ? p.copy({ mode: 'detailed', currentTrackId: trackId }) : p))
            const play = !!player?.isPlaying
            seekTo(startTimeSec, play)
        },
        [player?.isPlaying, seekTo]
    )

    const goToAdjacentTrack = React.useCallback(
        (forward: boolean) => {
            if (!player) return
            const tracks = getSortedTracks()
            if (tracks.length === 0) return
            const currentTime = player.currentTimeSec
            if (player.mode === 'detailed' && player.currentTrackId != null) {
                const idx = tracks.findIndex(t => t.track_id === player.currentTrackId)
                if (idx < 0) return
                const nextIdx = (idx + (forward ? 1 : -1) + tracks.length) % tracks.length
                const t = tracks[nextIdx]
                goToTrack(t.track_id, t.start_time_seconds)
            } else {
                if (forward) {
                    const t = tracks.find(t => t.start_time_seconds > currentTime) ?? tracks[0]
                    goToTrack(t.track_id, t.start_time_seconds)
                } else {
                    const t =
                        [...tracks].reverse().find(t => t.start_time_seconds < currentTime) ?? tracks[tracks.length - 1]
                    goToTrack(t.track_id, t.start_time_seconds)
                }
            }
        },
        [player, getSortedTracks, goToTrack]
    )

    // Keyboard controls
    React.useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            // prevent interfering when typing
            const target = e.target as HTMLElement | null
            if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable))
                return
            // Avoid any user interactions while export temporarily controls playback/seek.
            if (isExporting) return
            if (!player) return
            const key = e.key.length === 1 ? e.key.toLowerCase() : e.key
            if (key === ' ') {
                e.preventDefault()
                const v = videoRef.current
                if (v && v.duration && v.currentTime >= v.duration - 0.05) {
                    trackEvent('shortcut_used', { action: 'restart_play' })
                    seekTo(0, true)
                } else {
                    trackEvent('shortcut_used', { action: 'toggle_play' })
                    togglePlay()
                }
            } else if (key === 'ArrowLeft' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'step_prev_frame' })
                stepPrev()
            } else if (key === 'ArrowRight' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'step_next_frame' })
                stepNext()
            } else if (e.ctrlKey && key === 'ArrowLeft') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'seek_minus_30s' })
                seekTo(player.currentTimeSec - 30, true)
            } else if (e.ctrlKey && key === 'ArrowRight') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'seek_plus_30s' })
                seekTo(player.currentTimeSec + 30, true)
            } else if (e.shiftKey && key === 'ArrowLeft') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'seek_minus_5s' })
                seekTo(player.currentTimeSec - 5, true)
            } else if (e.shiftKey && key === 'ArrowRight') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'seek_plus_5s' })
                seekTo(player.currentTimeSec + 5, true)
            } else if (key === '-') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'speed_down' })
                bumpSpeed(true)
            } else if (key === '+' || key === '=') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'speed_up' })
                bumpSpeed(false)
            } else if (key.toLowerCase() === 'escape') {
                if (player.mode === 'overview') {
                    trackEvent('shortcut_used', { action: 'close_player' })
                    onClose?.()
                } else {
                    trackEvent('shortcut_used', { action: 'exit_detailed_view' })
                    setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))
                }
            } else if (!e.shiftKey && key.toLowerCase() === 'n') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'next_track' })
                goToAdjacentTrack(true)
            } else if (!e.shiftKey && key.toLowerCase() === 'p') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'prev_track' })
                goToAdjacentTrack(false)
            } else if (e.shiftKey && key.toLowerCase() === 'n') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'next_video' })
                onOpenNextJob?.()
            } else if (e.shiftKey && key.toLowerCase() === 'p') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'prev_video' })
                onOpenPrevJob?.()
            }
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [
        player,
        isExporting,
        togglePlay,
        stepPrev,
        stepNext,
        seekTo,
        bumpSpeed,
        goToAdjacentTrack,
        onClose,
        onOpenNextJob,
        onOpenPrevJob,
    ])

    // Video frame callback -> sync currentTimeSec + draw
    React.useEffect(() => {
        const v = videoRef.current
        const c = canvasRef.current
        const container = containerRef.current
        if (!v || !c || !player || !container) return
        // During export we drive the decoder timeline ourselves; avoid mutating player state while exporting.
        if (isExporting) return

        let vfId: number | null = null
        const onFrame = (_: number, meta: VideoFrameCallbackMetadata) => {
            const nowSec = meta.mediaTime
            setPlayer(prev => (prev ? prev.copy({ currentTimeSec: nowSec }) : prev))
            // Draw the current frame immediately for smooth playback
            drawFrame(
                c,
                container,
                v,
                player,
                { zoom, offsetX: offset.x, offsetY: offset.y, hoveredTrackId },
                nowSec,
                job.dominant_orientation
            )
            vfId = v.requestVideoFrameCallback(onFrame)
        }
        vfId = v.requestVideoFrameCallback(onFrame)
        return () => {
            if (vfId) v.cancelVideoFrameCallback(vfId)
        }
    }, [
        isExporting,
        player?.isPlaying,
        player?.mode,
        player?.currentTrackId,
        zoom,
        offset.x,
        offset.y,
        hoveredTrackId,
        videoUrl,
    ])

    // Redraw on resize or when paused and state changes
    React.useEffect(() => {
        const c = canvasRef.current
        const v = videoRef.current
        const container = containerRef.current
        if (!c || !v || !player || !container) return
        if (isExporting) return
        drawFrame(
            c,
            container,
            v,
            player,
            { zoom, offsetX: offset.x, offsetY: offset.y, hoveredTrackId },
            undefined,
            job.dominant_orientation
        )
    }, [
        isExporting,
        player?.currentTimeSec,
        player?.mode,
        player?.currentTrackId,
        zoom,
        offset.x,
        offset.y,
        hoveredTrackId,
        job.dominant_orientation,
    ])

    // Auto-exit detailed mode if no reasonably recent detection around current time
    React.useEffect(() => {
        if (isExporting) return
        if (!player || player.mode !== 'detailed' || player.currentTrackId == null) return
        if (!player.hasDetectionAfter(player.currentTrackId, player.currentTimeSec)) {
            setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))
        }
    }, [isExporting, player?.currentTimeSec, player?.mode, player?.currentTrackId])

    // bumpSpeed is provided by usePlaybackSpeed

    const onWheelCanvas = React.useCallback(
        (e: React.WheelEvent<HTMLCanvasElement>) => {
            if (!player || player.mode !== 'overview') return
            const container = containerRef.current
            const rect = (container ?? e.currentTarget).getBoundingClientRect()
            const px = e.clientX - rect.left
            const py = e.clientY - rect.top
            // Pass coordinates relative to the container center to keep the cursor point anchored
            const absX = rect.left + (px - rect.width * 0.5)
            const absY = rect.top + (py - rect.height * 0.5)
            onWheelZoom(absX, absY, e.deltaY)
        },
        [player, onWheelZoom]
    )

    const onMouseMove = React.useCallback(
        (e: React.MouseEvent<HTMLCanvasElement>) => {
            if (!player || player.mode !== 'overview') return
            const c = canvasRef.current
            if (!c) return
            const container = containerRef.current
            const rect = (container ?? c).getBoundingClientRect()
            const px = e.clientX - rect.left
            const py = e.clientY - rect.top
            const hit = pickTrackAtScreenPoint(
                px,
                py,
                rect.width,
                rect.height,
                player,
                {
                    zoom,
                    offsetX: offset.x,
                    offsetY: offset.y,
                    hoveredTrackId,
                },
                job.dominant_orientation
            )
            setHoveredTrackId(hit)
        },
        [player, zoom, offset.x, offset.y, hoveredTrackId, job.dominant_orientation]
    )

    const onClick = React.useCallback(() => {
        setPlayer(p => {
            if (!p) return p
            if (p.mode !== 'overview' || hoveredTrackId == null)
                return p.copy({ mode: 'overview', currentTrackId: null })

            const trackId = hoveredTrackId
            trackEvent('surfer_clicked', { track_id: trackId })
            const detection = p.interpolateDetectionByTime(trackId, p.currentTimeSec)
            if (!detection) return p.copy({ mode: 'overview', currentTrackId: null })

            const t = detection.time_percent * p.video.durationSeconds
            videoRef.current!.currentTime = t
            return p.copy({ mode: 'detailed', currentTrackId: trackId, currentTimeSec: t })
        })
    }, [hoveredTrackId])

    const exportVisible = !!player && player.mode === 'detailed' && player.currentTrackId != null
    const exportEnabled = exportVisible && !isExporting

    const onExportTrack = React.useCallback(async () => {
        if (isExporting) return
        setExportError(null)
        setIsExporting(true)
        setExportProgressPct(0)

        try {
            const file = sourceFile
            const p = player
            if (!file || !p) throw new Error('Not ready')
            if (p.mode !== 'detailed' || p.currentTrackId == null) throw new Error('Select a track')

            const track = p.tracks.find(t => t.track_id === p.currentTrackId)
            if (!track) throw new Error('Track not found')
            trackEvent('export_track_start', { job_id: job.id, track_id: track.track_id })

            const padSec = 0.25
            const startSec = Math.max(0, track.start_time_seconds - padSec)
            const endSec = Math.min(
                p.video.durationSeconds || Infinity,
                track.start_time_seconds + track.duration_seconds + padSec
            )
            if (!(endSec > startSec + 1e-3)) throw new Error('Track duration too short')

            const outBlob = await exportTrackMp4({
                file,
                player: p,
                dominantOrientationDeg: job.dominant_orientation,
                trackId: track.track_id,
                startSec,
                endSec,
                onProgress: prog01 => setExportProgressPct(clamp(prog01 * 100, 0, 100)),
            })

            const filename = buildExportFilename({
                sourceFileName: sourceFile?.name,
                localRelativePath: job.local_relative_path,
                trackId: track.track_id,
                startSec,
                endSec,
            })

            downloadBlob(outBlob, filename)
            trackEvent('export_track_success', { job_id: job.id, track_id: track.track_id })
        } catch (e: any) {
            setExportError(String(e?.message || e || 'Export failed'))
            trackEvent('export_track_failed', { job_id: job.id, message: String(e?.message || e || 'Export failed') })
        } finally {
            setIsExporting(false)
            setExportProgressPct(null)
        }
    }, [isExporting, job.dominant_orientation, job.id, job.local_relative_path, player, sourceFile])

    return (
        <div className="relative flex flex-col h-full">
            <div className="relative flex-1 bg-black overflow-hidden">
                {error && <div className="absolute left-2 top-2 text-red-500 text-sm">{error}</div>}
                {exportError && <div className="absolute left-2 top-7 text-red-400 text-sm">Export: {exportError}</div>}
                {fileMissing && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-red-500 text-3xl font-extrabold tracking-wide">VIDEO FILE NOT FOUND</div>
                    </div>
                )}
                <div ref={containerRef} className="absolute inset-0">
                    <canvas
                        ref={canvasRef}
                        className="absolute inset-0 block"
                        onWheel={onWheelCanvas}
                        onMouseMove={onMouseMove}
                        onMouseLeave={() => setHoveredTrackId(null)}
                        onClick={onClick}
                    />
                    {/* Hidden video used only for decoding frames */}
                    {videoUrl && (
                        <video
                            ref={videoRef}
                            key={videoUrl}
                            src={videoUrl}
                            playsInline
                            muted={true}
                            autoPlay={true}
                            preload="metadata"
                            style={{ width: 0, height: 0, opacity: 0, position: 'absolute' }}
                            onEnded={() => {
                                setPlayer(p => (p ? p.copy({ isPlaying: false }) : p))
                                videoRef.current?.pause()
                            }}
                        />
                    )}
                </div>
            </div>

            {player && (
                <div className="px-3 py-2 bg-black/60 border-t border-gray-700">
                    <div className="mb-2">
                        <Timeline state={player} onSeekTime={t => seekTo(t, false)} />
                    </div>
                    <ControlsBar
                        onPlayPause={() => {
                            trackEvent('player_play_pause_clicked', { job_id: job.id })
                            togglePlay()
                        }}
                        onSpeedDown={() => {
                            trackEvent('player_speed_down_clicked', { job_id: job.id })
                            bumpSpeed(true)
                        }}
                        onSpeedUp={() => {
                            trackEvent('player_speed_up_clicked', { job_id: job.id })
                            bumpSpeed(false)
                        }}
                        isPlaying={player.isPlaying}
                        speed={speed}
                        zoom={zoom}
                        onExportTrack={onExportTrack}
                        exportVisible={exportVisible}
                        exportEnabled={exportEnabled}
                        isExporting={isExporting}
                        exportProgressPct={exportProgressPct}
                    />
                </div>
            )}

            {/* Blocking overlay during export (covers canvas + controls, captures all pointer interactions) */}
            {isExporting && (
                <div
                    className="absolute inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
                    onMouseDown={e => e.preventDefault()}
                    onClick={e => e.preventDefault()}
                    onWheel={e => e.preventDefault()}
                >
                    <div className="px-4 py-3 rounded-lg bg-black/60 border border-gray-700 text-gray-100 text-center">
                        <div className="text-base font-semibold">Exporting track…</div>
                        {typeof exportProgressPct === 'number' ? (
                            <div className="mt-1 text-sm tabular-nums">
                                {Math.max(0, Math.min(100, exportProgressPct)).toFixed(0)}%
                            </div>
                        ) : (
                            <div className="mt-1 text-sm">Starting…</div>
                        )}
                        <div className="mt-2 text-xs text-gray-300">
                            Please wait — playback controls are temporarily disabled.
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

// Helpers
let _sharedOffscreenCanvas: HTMLCanvasElement | null = null
function getSharedOffscreenCanvas(): HTMLCanvasElement {
    if (!_sharedOffscreenCanvas) {
        _sharedOffscreenCanvas = document.createElement('canvas')
    }
    return _sharedOffscreenCanvas
}

function ensureCanvasSize(canvas: HTMLCanvasElement, cssWidth: number, cssHeight: number) {
    const dpr = Math.max(1, Math.floor(window.devicePixelRatio))
    const needW = Math.max(1, Math.floor(cssWidth * dpr))
    const needH = Math.max(1, Math.floor(cssHeight * dpr))
    if (canvas.width !== needW || canvas.height !== needH) {
        canvas.width = needW
        canvas.height = needH
    }
    canvas.style.width = `${cssWidth}px`
    canvas.style.height = `${cssHeight}px`
    const ctx = canvas.getContext('2d')!
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.scale(dpr, dpr)
    return ctx
}

function computeBaseRect(outW: number, outH: number, vidW: number, vidH: number) {
    const scale = Math.min(outW / vidW, outH / vidH)
    const dispW = vidW * scale
    const dispH = vidH * scale
    const offX = (outW - dispW) / 2
    const offY = (outH - dispH) / 2
    return { x: offX, y: offY, w: dispW, h: dispH, scale }
}

function drawFitContain(ctx: Ctx2D, outW: number, outH: number, src: CanvasImageSource, srcW: number, srcH: number) {
    const base = computeBaseRect(outW, outH, srcW, srcH)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(src, 0, 0, srcW, srcH, base.x, base.y, base.w, base.h)
}

function drawDetailedCrop(
    ctx: Ctx2D,
    outputWidth: number,
    outputHeight: number,
    srcCanvas: CanvasImageSource,
    srcWidth: number,
    srcHeight: number,
    det: TimedBBox | null
) {
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, outputWidth, outputHeight)

    if (!det) {
        drawFitContain(ctx, outputWidth, outputHeight, srcCanvas, srcWidth, srcHeight)
        return
    }

    const [x1p, y1p, x2p, y2p] = det.bbox
    const x1 = x1p * srcWidth
    const y1 = y1p * srcHeight
    const x2 = x2p * srcWidth
    const y2 = y2p * srcHeight
    const bboxW = Math.max(1, x2 - x1)
    const bboxH = Math.max(1, y2 - y1)

    const sHeight = (TARGET_BBOX_HEIGHT_RATIO * outputHeight) / bboxH
    const sWidthLimit = outputWidth / bboxW
    const s = clamp(Math.min(sHeight, sWidthLimit), MIN_SCALE, MAX_SCALE)

    const cx = (x1 + x2) * 0.5
    const cy = (y1 + y2) * 0.5
    const cropW = outputWidth / s
    const cropH = outputHeight / s
    const winX1 = cx - cropW / 2
    const winY1 = cy - cropH / 2
    const winX2 = winX1 + cropW
    const winY2 = winY1 + cropH

    const srcX1 = Math.max(0, Math.floor(winX1))
    const srcY1 = Math.max(0, Math.floor(winY1))
    const srcX2 = Math.min(srcWidth, Math.ceil(winX2))
    const srcY2 = Math.min(srcHeight, Math.ceil(winY2))
    const dstX1 = Math.max(0, Math.floor((srcX1 - winX1) * s))
    const dstY1 = Math.max(0, Math.floor((srcY1 - winY1) * s))
    const dstX2 = Math.min(outputWidth, Math.ceil((srcX2 - winX1) * s))
    const dstY2 = Math.min(outputHeight, Math.ceil((srcY2 - winY1) * s))

    const srcWW = clamp(srcX2 - srcX1, 0, srcWidth)
    const srcHH = clamp(srcY2 - srcY1, 0, srcHeight)
    const dstWW = clamp(dstX2 - dstX1, 0, outputWidth)
    const dstHH = clamp(dstY2 - dstY1, 0, outputHeight)

    if (srcWW > 0 && srcHH > 0 && dstWW > 0 && dstHH > 0) {
        ctx.imageSmoothingEnabled = true
        ctx.imageSmoothingQuality = 'high'
        ctx.drawImage(srcCanvas, srcX1, srcY1, srcWW, srcHH, dstX1, dstY1, dstWW, dstHH)
    }
}

function drawFrame(
    canvas: HTMLCanvasElement,
    containerEl: HTMLElement,
    video: HTMLVideoElement,
    player: PlayerState,
    ov: OverviewView,
    timeOverrideSec?: number,
    dominantOrientationDeg: number = 0
) {
    const rect = containerEl.getBoundingClientRect()
    const cssW = Math.max(1, Math.floor(rect.width))
    const cssH = Math.max(1, Math.floor(rect.height))
    const ctx = ensureCanvasSize(canvas, cssW, cssH)

    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, cssW, cssH)

    if (!video.videoWidth || !video.videoHeight) return

    // Prepare source draw surface with orientation applied (reuse a single offscreen canvas)
    const offscreen = getSharedOffscreenCanvas()
    const rotatedVideo = drawRotatedToCanvas(video, offscreen, dominantOrientationDeg)

    // Current time for stabilization lookup
    const now = timeOverrideSec ?? player.currentTimeSec
    let sourceCanvas: HTMLCanvasElement = offscreen

    if (player.mode === 'overview') {
        const base = computeBaseRect(cssW, cssH, rotatedVideo.width, rotatedVideo.height)
        const z = ov.zoom
        const cx = base.x + base.w * 0.5 + ov.offsetX
        const cy = base.y + base.h * 0.5 + ov.offsetY
        const sBase = (base.w / rotatedVideo.width) * z
        const stab = player.getStabilizationAt(now)

        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(sBase, sBase)
        // Apply cumulative stabilization as pre-apply transform: translate then rotate
        ctx.translate(stab.dx, stab.dy)
        ctx.rotate(stab.da)
        ctx.imageSmoothingEnabled = true
        ctx.imageSmoothingQuality = 'high'
        ctx.drawImage(offscreen, -rotatedVideo.width * 0.5, -rotatedVideo.height * 0.5)

        if (false) {
            drawStabilizationTransforms(ctx, player, now, sBase, cx, cy)
        }

        // Draw detections under same transform
        for (const t of player.tracks) {
            const isHovered = ov.hoveredTrackId === t.track_id
            if (!isHovered) continue

            const det = player.interpolateDetectionByTime(t.track_id, now)
            if (!det) continue
            const [x1p, y1p, x2p, y2p] = det.bbox
            const x1 = x1p * rotatedVideo.width - rotatedVideo.width * 0.5
            const y1 = y1p * rotatedVideo.height - rotatedVideo.height * 0.5
            const w = Math.max(1, (x2p - x1p) * rotatedVideo.width)
            const h = Math.max(1, (y2p - y1p) * rotatedVideo.height)

            ctx.strokeStyle = '#10b981'
            ctx.lineWidth = 2 / sBase
            ctx.strokeRect(Math.round(x1) + 0.5, Math.round(y1) + 0.5, Math.round(w), Math.round(h))
        }
        ctx.restore()
    } else if (player.mode === 'detailed' && player.currentTrackId != null) {
        const now = timeOverrideSec ?? player.currentTimeSec
        const det = player.interpolateDetectionByTime(player.currentTrackId, now)
        if (!det) return

        const vidW = rotatedVideo.width
        const vidH = rotatedVideo.height
        // Reuse shared crop-draw logic (same as export path).
        const detTimed: TimedBBox = { time_percent: det.time_percent, bbox: det.bbox }
        drawDetailedCrop(ctx, cssW, cssH, sourceCanvas, vidW, vidH, detTimed)
    }
}

function drawText(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, bgColor: string) {
    ctx.fillStyle = bgColor
    ctx.font = '12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto'
    ctx.fillRect(x, y, ctx.measureText(text).width, 14)
    ctx.fillStyle = color
    ctx.fillText(text, x + 3, y + 11)
}

function pickTrackAtScreenPoint(
    px: number,
    py: number,
    outW: number,
    outH: number,
    player: PlayerState,
    ov: OverviewView,
    dominantOrientationDeg: number = 0
): number | null {
    const { width, height } = getRotatedDimensions(player.video.width, player.video.height, dominantOrientationDeg)
    const base = computeBaseRect(outW, outH, width, height)
    const stab = player.getStabilizationAt(player.currentTimeSec)

    const z = ov.zoom
    const cx = base.x + base.w * 0.5 + ov.offsetX
    const cy = base.y + base.h * 0.5 + ov.offsetY
    const sBase = (base.w / width) * z

    // Undo transform chain: Translate(cx,cy) -> Scale(s) -> Translate(stab) -> Rotate(stab)

    // 1. Undo Screen Center Translation
    const dx0 = px - cx
    const dy0 = py - cy

    // 2. Undo Scale
    const dx1 = dx0 / sBase
    const dy1 = dy0 / sBase

    // 3. Undo Stabilization Translation
    const dx2 = dx1 - stab.dx
    const dy2 = dy1 - stab.dy

    // 4. Undo Stabilization Rotation
    const cos = Math.cos(-stab.da)
    const sin = Math.sin(-stab.da)
    const dx3 = dx2 * cos - dy2 * sin
    const dy3 = dx2 * sin + dy2 * cos

    // Convert from centered coords to normalized [0,1]
    const xNorm = (dx3 + width * 0.5) / width
    const yNorm = (dy3 + height * 0.5) / height

    for (const t of player.tracks) {
        const det = player.interpolateDetectionByTime(t.track_id, player.currentTimeSec)
        if (!det) continue
        const [x1p, y1p, x2p, y2p] = det.bbox

        if (xNorm >= x1p && xNorm <= x2p && yNorm >= y1p && yNorm <= y2p) {
            return t.track_id
        }
    }
    return null
}

function drawStabilizationTransforms(
    ctx: CanvasRenderingContext2D,
    player: PlayerState,
    now: number,
    sBase: number,
    cx: number,
    cy: number
) {
    // Debug: draw stabilization trail (last ~30 samples) anchored at center (relative to current)
    try {
        const N = 30
        const dt = 1 / 30 // seconds per sample
        const sScale = sBase
        const pts: Array<{ x: number; y: number }> = []
        const siNow = player.getStabilizationAt(now)
        const vx0 = sScale * siNow.dx
        const vy0 = sScale * siNow.dy
        for (let i = 0; i < N; i++) {
            const t = Math.max(0, now - i * dt)
            const si = player.getStabilizationAt(t)
            const vx = sScale * si.dx
            const vy = sScale * si.dy
            const px = cx + (vx - vx0) - 300
            const py = cy + (vy - vy0) - 300
            pts.push({ x: px, y: py })
        }
        if (pts.length >= 2) {
            ctx.save()
            ctx.lineWidth = 2
            for (let i = 0; i < pts.length - 1; i++) {
                const a = Math.max(0.15, 1 - i / pts.length)
                ctx.strokeStyle = `rgba(56,189,248,${a})` // cyan-400 with fade
                ctx.beginPath()
                ctx.moveTo(pts[i].x, pts[i].y)
                ctx.lineTo(pts[i + 1].x, pts[i + 1].y)
                ctx.stroke()
            }
            // head marker
            ctx.fillStyle = 'rgba(56,189,248,0.9)'
            ctx.beginPath()
            ctx.arc(cx, cy, 3, 0, Math.PI * 2)
            ctx.fill()
            ctx.restore()
        }
    } catch {}
}

function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.style.display = 'none'
    document.body.appendChild(a)
    a.click()
    a.remove()
    // Give the download a moment to start before revoking.
    setTimeout(() => URL.revokeObjectURL(url), 1000)
}

type WatermarkAsset = { img: CanvasImageSource; width: number; height: number }

async function getWatermarkAsset(): Promise<WatermarkAsset | null> {
    try {
        // Respect Vite base path if present (e.g. hosted under a subpath).
        const url = new URL(`logo.png`, globalThis.location?.href).toString()

        // Prefer ImageBitmap (works with both canvas types).
        if (typeof fetch === 'function' && typeof createImageBitmap === 'function') {
            const res = await fetch(url, { cache: 'force-cache' })
            if (!res.ok) throw new Error(`Failed to load watermark (${res.status})`)
            const blob = await res.blob()
            const bmp = await createImageBitmap(blob)
            return { img: bmp, width: bmp.width, height: bmp.height }
        }

        // Fallback for environments without createImageBitmap.
        if (typeof document !== 'undefined') {
            const img = new Image()
            img.decoding = 'async'
            img.crossOrigin = 'anonymous'
            img.src = url
            await img.decode()
            return { img, width: img.naturalWidth || img.width, height: img.naturalHeight || img.height }
        }
    } catch {
        // Watermark is best-effort; export should still succeed.
    }
    return null
}

function drawWatermark(
    ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D,
    outW: number,
    outH: number,
    wm: WatermarkAsset | null
) {
    if (!wm || wm.width <= 0 || wm.height <= 0) return

    // Keep it unobtrusive: cap size and never upscale.
    const pad = Math.round(outW * 0.02)
    const maxW = Math.min(240, Math.round(outW * 0.22))
    const maxH = Math.min(80, Math.round(outH * 0.18))
    const s = Math.min(maxW / wm.width, maxH / wm.height, 1)
    const w = Math.max(1, Math.round(wm.width * s))
    const h = Math.max(1, Math.round(wm.height * s))

    const x = outW - pad - w
    const y = outH - pad - h

    ctx.save()
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.globalAlpha = 0.65

    // Subtle shadow helps readability on bright backgrounds.
    ctx.shadowColor = 'rgba(0,0,0,0.35)'
    ctx.shadowBlur = 6
    ctx.shadowOffsetX = 0
    ctx.shadowOffsetY = 2

    ctx.drawImage(wm.img, x, y, w, h)
    ctx.restore()
}

function sanitizeFilenameBase(name: string): string {
    // Basic Windows/macOS-friendly sanitization.
    const cleaned = name.replace(/[<>:"/\\|?*\x00-\x1F]+/g, '_').trim()
    return cleaned.length > 0 ? cleaned : 'export'
}

function basename(path: string): string {
    const parts = path.split(/[\\/]+/).filter(Boolean)
    return parts.length ? parts[parts.length - 1] : path
}

function stripExtension(name: string): string {
    return name.replace(/\.[^./\\]+$/, '')
}

function buildExportFilename(params: {
    sourceFileName: string | null
    localRelativePath: string | null | undefined
    trackId: number
    startSec: number
    endSec: number
}): string {
    const baseFromFile = params.sourceFileName ? stripExtension(basename(params.sourceFileName)) : ''
    const baseFromPath = params.localRelativePath ? stripExtension(basename(params.localRelativePath)) : ''
    const base = sanitizeFilenameBase(baseFromFile || baseFromPath)
    const start = params.startSec.toFixed(2)
    const end = params.endSec.toFixed(2)
    return `${base}_track_${params.trackId}_${start}-${end}.mp4`
}

async function exportTrackMp4(params: {
    file: File
    player: PlayerState
    dominantOrientationDeg: number
    trackId: number
    startSec: number
    endSec: number
    onProgress?: (p01: number) => void
}): Promise<Blob> {
    const { file, player, dominantOrientationDeg, trackId, startSec, endSec, onProgress } = params

    const outputWidth = 1280
    const outputHeight = 720
    const bitrate = 2_000_000

    // Best-effort watermark; if it fails to load, we still export.
    const watermark = await getWatermarkAsset()

    const onFrame = async (frame: VideoFrame, ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D) => {
        const tSec = (frame.timestamp || 0) / 1_000_000

        if (tSec + 1e-6 < startSec) return false
        if (tSec >= endSec) return 'stop'

        const rotCanvas = getSharedOffscreenCanvas()
        const rotated = drawRotatedToCanvas(frame, rotCanvas, dominantOrientationDeg)

        const det = player.interpolateDetectionByTime(trackId, tSec)
        drawDetailedCrop(ctx, outputWidth, outputHeight, rotCanvas, rotated.width, rotated.height, det)
        drawWatermark(ctx, outputWidth, outputHeight, watermark)
        return true
    }

    const outBuf = await processVideo({ file, onFrame, outputWidth, outputHeight, bitrate, onProgress })
    return new Blob([outBuf], { type: 'video/mp4' })
}
