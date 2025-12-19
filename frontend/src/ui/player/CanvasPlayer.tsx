import React from 'react'
import { JobDetail, ReportType } from '../types'
import { PlayerState, VideoProperties } from './state'
import { ControlsBar } from './ControlsBar'
import { Timeline } from './Timeline'
import { useZoom } from '../hooks/useZoom'
import { usePlaybackSpeed } from '../hooks/usePlaybackSpeed'
import { useSeeker } from '../hooks/useSeeker'
import { clamp } from '../utils/clamp'
import { trackEvent } from '../utils/analytics'
import { buildExportFilename, downloadExport, exportTrackMp4 } from './export'
import { useJobVideoSource } from './useJobVideoSource'
import { drawFrame, pickTrackAtScreenPoint } from './rendering'

type Props = {
    job: JobDetail
    dirHandle: FileSystemDirectoryHandle | null
    onClose: () => void
    onDelete: (id: string) => void
    onReport: (id: string, type: ReportType, message: string) => void
    onOpenNextJob?: () => void
    onOpenPrevJob?: () => void
}

export const CanvasPlayer: React.FC<Props> = ({ job, dirHandle, onClose, onOpenNextJob, onOpenPrevJob }) => {
    const [exportError, setExportError] = React.useState<string | null>(null)
    const videoRef = React.useRef<HTMLVideoElement | null>(null)
    const containerRef = React.useRef<HTMLDivElement | null>(null)
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
    const [player, setPlayer] = React.useState<PlayerState | null>(null)
    const { zoom, offset, onWheelZoom } = useZoom(containerRef)
    const { speed, bumpSpeed } = usePlaybackSpeed(1.0)
    const [hoveredTrackId, setHoveredTrackId] = React.useState<number | null>(null)
    const [isExporting, setIsExporting] = React.useState<boolean>(false)
    const [exportProgressPct, setExportProgressPct] = React.useState<number | null>(null)

    // Seeker (frame stepping and seeking)
    const { seekTo, stepNext, stepPrev, onNewFile } = useSeeker(videoRef, player, setPlayer)

    const { videoUrl, sourceFile, fileMissing, error } = useJobVideoSource({
        job,
        dirHandle,
        onFileLoaded: onNewFile,
    })

    React.useEffect(() => {
        trackEvent('player_open', { job_id: job.id })
        setExportError(null)
        setIsExporting(false)
        setExportProgressPct(null)
    }, [job.id])

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

            downloadExport(outBlob, filename)
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
