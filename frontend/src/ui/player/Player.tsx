import React from 'react'
import { useTranslation } from 'react-i18next'
import { JobDetail, ReportType, Track } from '../types'
import { PlayerState } from './state'
import { ControlsBar } from './ControlsBar'
import { Timeline } from './Timeline'
import { useZoom } from '../hooks/useZoom'
import { useCappedValue } from '../hooks/useCappedValue'
import { usePlaybackSpeed } from '../hooks/usePlaybackSpeed'
import { clamp } from '../utils/clamp'
import { trackEvent } from '../utils/analytics'
import { loadSetting, saveSetting } from '../utils/idb'
import { buildExportFilename, downloadExport, exportTrackMp4 } from './export'
import { useJobVideoSource } from './useJobVideoSource'
import { drawFrame, pickTrackAtScreenPoint, screenPointToVideoNorm } from './rendering'
import { useAnnotations } from './useAnnotations'
import { useWebCodexPlayer } from './useWebCodexPlayer'

const PLAYER_FOCUSED_CLICK_HINT_DISMISSED_KEY = 'player.focusedClickHintDismissed.v1'

type Props = {
    job: JobDetail
    dirHandle: FileSystemDirectoryHandle | null
    onClose: () => void
    onReport: (id: string, type: ReportType, message: string) => void
    onOpenNextJob?: () => void
    onOpenPrevJob?: () => void
    drawMode: boolean
    onToggleDrawMode: () => void
    player: PlayerState | null
    setPlayer: React.Dispatch<React.SetStateAction<PlayerState | null>>
}

export const Player: React.FC<Props> = ({
    job,
    dirHandle,
    onClose,
    onOpenNextJob,
    onOpenPrevJob,
    drawMode,
    onToggleDrawMode,
    player,
    setPlayer,
}) => {
    const { t } = useTranslation()
    const [exportError, setExportError] = React.useState<string | null>(null)
    const [playerInitError, setPlayerInitError] = React.useState<string | null>(null)
    const containerRef = React.useRef<HTMLDivElement | null>(null)
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
    const { zoom, offset, onWheelZoom } = useZoom({ minZoom: 0.5, maxZoom: 4 })
    const detailedZoom = useCappedValue(1, 0.5, 2.0)
    const { speed, bumpSpeed } = usePlaybackSpeed(1.0)
    const [hoveredTrackId, setHoveredTrackId] = React.useState<number | null>(null)
    const [isExporting, setIsExporting] = React.useState<boolean>(false)
    const [exportProgressPct, setExportProgressPct] = React.useState<number | null>(null)
    const [focusedClickHintDismissed, setFocusedClickHintDismissed] = React.useState<boolean>(false)

    const { sourceFile, fileMissing, error } = useJobVideoSource({ job, dirHandle })
    const errorText = error ? t(error.key, { message: error.detail }) : null

    const webPlayer = useWebCodexPlayer({ playbackRate: speed })

    const seekToFrame = React.useCallback(
        (frameIndex: number, play: boolean) => {
            const n = webPlayer.frameCount
            if (!webPlayer.ready || n <= 0) return
            const idx = clamp(frameIndex, 0, n - 1)
            void webPlayer.seekFrame(idx, play)
        },
        [webPlayer.ready, webPlayer.frameCount, webPlayer.seekFrame]
    )

    const getDrawPoint = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            if (!player) return null
            const c = canvasRef.current
            const container = containerRef.current
            const rect = (container ?? c)?.getBoundingClientRect()
            if (!rect) return null
            const px = e.clientX - rect.left
            const py = e.clientY - rect.top
            return screenPointToVideoNorm(
                px,
                py,
                rect.width,
                rect.height,
                player,
                webPlayer.width,
                webPlayer.height,
                webPlayer.currentFrameIndex,
                {
                    zoom,
                    detailedZoom: detailedZoom.value,
                    offsetX: offset.x,
                    offsetY: offset.y,
                    hoveredTrackId,
                },
                job.dominant_orientation
            )
        },
        [
            player,
            zoom,
            detailedZoom.value,
            offset.x,
            offset.y,
            hoveredTrackId,
            job.dominant_orientation,
            webPlayer.currentFrameIndex,
        ]
    )

    const annotations = useAnnotations(getDrawPoint, {
        drawMode,
        isExporting,
        isPlaying: webPlayer.playing,
        currentFrameIndex: webPlayer.currentFrameIndex,
    })

    React.useEffect(() => {
        loadSetting<boolean>(PLAYER_FOCUSED_CLICK_HINT_DISMISSED_KEY).then(saved => {
            setFocusedClickHintDismissed(!!saved)
        })
    }, [])

    React.useEffect(() => {
        trackEvent('player_open', { job_id: job.id })
        setExportError(null)
        setPlayerInitError(null)
        setIsExporting(false)
        setExportProgressPct(null)
        detailedZoom.reset()
        annotations.reset()
    }, [job.id])

    React.useEffect(() => {
        // reset detailed zoom when leaving detailed mode
        if (player?.mode !== 'detailed') {
            detailedZoom.reset()
        }
    }, [player?.mode])

    React.useEffect(() => {
        async function start() {
            const mySourceFile = sourceFile
            setPlayer(null)
            await webPlayer.dispose()
            if (!mySourceFile || mySourceFile !== sourceFile) return
            await webPlayer.load(mySourceFile)
            webPlayer.play()
        }
        start()
    }, [sourceFile])

    React.useEffect(() => {
        if (!webPlayer.ready) return
        try {
            setPlayer(PlayerState.from(job, webPlayer.frameCount))
        } catch (e: any) {
            setPlayer(null)
            setPlayerInitError(String(e?.message ?? e ?? 'Failed to initialize player state'))
        }
    }, [job, webPlayer.ready, webPlayer.frameCount])

    React.useEffect(() => {
        if (!webPlayer.ready) return
        if (isExporting) {
            webPlayer.pause()
            return
        }
    }, [isExporting, webPlayer.ready])

    const handlePlayPause = React.useCallback(() => {
        if (drawMode) {
            onToggleDrawMode()
            if (webPlayer.ready) webPlayer.play()
            return
        }
        if (!webPlayer.ready) return
        if (webPlayer.playing) {
            webPlayer.pause()
            return
        }
        if (webPlayer.ended) {
            seekToFrame(0, true)
            return
        }
        webPlayer.play()
    }, [
        drawMode,
        onToggleDrawMode,
        webPlayer.ready,
        webPlayer.playing,
        webPlayer.play,
        webPlayer.pause,
        webPlayer.ended,
        seekToFrame,
    ])

    const goToAdjacentTrack = React.useCallback(
        (forward: boolean) => {
            const goToTrack = (track: Track) => {
                const n = webPlayer.frameCount
                if (!player || n <= 0) return
                const startFrame = player.frameIndexForPercent(track.start_percent)
                setPlayer(p => (p ? p.copy({ mode: 'detailed', currentTrackId: track.track_id }) : p))
                seekToFrame(startFrame, webPlayer.playing)
            }

            if (!player || player.tracks.length === 0 || webPlayer.frameCount <= 0) return
            const tracks = player.tracks
            if (player.mode === 'detailed' && player.currentTrackId != null) {
                const idx = tracks.findIndex(t => t.track_id === player.currentTrackId)
                if (idx < 0) return
                const nextIdx = (idx + (forward ? 1 : -1) + tracks.length) % tracks.length
                goToTrack(tracks[nextIdx])
            } else {
                const currentFrame = webPlayer.currentFrameIndex
                const n = webPlayer.frameCount
                if (forward) {
                    const track = tracks.find(t0 => player.frameIndexForPercent(t0.start_percent) > currentFrame)
                    goToTrack(track ?? tracks[0])
                } else {
                    const track = [...tracks]
                        .reverse()
                        .find(t0 => player.frameIndexForPercent(t0.start_percent) < currentFrame)
                    goToTrack(track ?? tracks[tracks.length - 1])
                }
            }
        },
        [player, webPlayer.frameCount, webPlayer.currentFrameIndex]
    )

    // Single draw point: re-render canvas after every React render.
    React.useLayoutEffect(() => {
        const c = canvasRef.current
        const container = containerRef.current
        const src = webPlayer.currentFrameCanvas
        if (!c || !container || !player || !src) return
        if (!webPlayer.ready) return
        if (isExporting) return

        const frameIndex = webPlayer.currentFrameIndex
        const visible = annotations.getVisibleAnnotations(frameIndex)

        drawFrame(
            c,
            container,
            src,
            { width: webPlayer.width, height: webPlayer.height },
            player,
            { zoom, detailedZoom: detailedZoom.value, offsetX: offset.x, offsetY: offset.y, hoveredTrackId },
            visible,
            frameIndex,
            job.dominant_orientation
        )
    })

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
                if (
                    webPlayer.ready &&
                    webPlayer.frameCount > 0 &&
                    webPlayer.currentFrameIndex >= webPlayer.frameCount - 1
                ) {
                    trackEvent('shortcut_used', { action: 'restart_play' })
                    seekToFrame(0, true)
                } else {
                    trackEvent('shortcut_used', { action: 'toggle_play' })
                    handlePlayPause()
                }
            } else if (key === 'ArrowLeft' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'step_prev_frame' })
                void webPlayer.stepFrames(-1)
            } else if (key === 'ArrowRight' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'step_next_frame' })
                void webPlayer.stepFrames(1)
            } else if (e.ctrlKey && key === 'ArrowLeft') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'seek_minus_30s' })
                seekToFrame(webPlayer.currentFrameIndex - 900, true)
            } else if (e.ctrlKey && key === 'ArrowRight') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'seek_plus_30s' })
                seekToFrame(webPlayer.currentFrameIndex + 900, true)
            } else if (e.shiftKey && key === 'ArrowLeft') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'seek_minus_5s' })
                seekToFrame(webPlayer.currentFrameIndex - 150, true)
            } else if (e.shiftKey && key === 'ArrowRight') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'seek_plus_5s' })
                seekToFrame(webPlayer.currentFrameIndex + 150, true)
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
            } else if (e.ctrlKey && key.toLowerCase() === 'z') {
                if (!drawMode) return
                if (webPlayer.playing) return
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'undo_draw' })
                annotations.undo()
            } else if (key.toLowerCase() === 'd') {
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'toggle_draw_mode' })
                onToggleDrawMode()
            }
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [
        player,
        isExporting,
        handlePlayPause,
        seekToFrame,
        bumpSpeed,
        goToAdjacentTrack,
        onClose,
        onOpenNextJob,
        onOpenPrevJob,
        onToggleDrawMode,
        drawMode,
        annotations,
    ])

    React.useEffect(() => {
        if (!drawMode) return
        setHoveredTrackId(null)
    }, [drawMode])

    React.useEffect(() => {
        if (!drawMode) return
        if (!webPlayer.playing) return
        webPlayer.pause()
    }, [drawMode, webPlayer.playing])

    // Auto-exit detailed mode if the track isn't active at the current frame.
    React.useEffect(() => {
        if (isExporting) return
        if (webPlayer.seeking || webPlayer.loading) return
        if (!player || player.mode !== 'detailed' || player.currentTrackId == null) return
        if (!player.isTrackActiveAtFrame(player.currentTrackId, webPlayer.currentFrameIndex)) {
            setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))
        }
    }, [
        isExporting,
        webPlayer.seeking,
        webPlayer.loading,
        webPlayer.currentFrameIndex,
        player?.mode,
        player?.currentTrackId,
    ])

    const onWheelCanvas = React.useCallback(
        (e: React.WheelEvent<HTMLCanvasElement>) => {
            if (drawMode) return
            if (!player) return
            e.preventDefault()

            switch (player.mode) {
                case 'detailed':
                    detailedZoom.set(z => z * (1 + (e.deltaY < 0 ? 0.1 : -0.1)))
                    break
                case 'overview':
                    const container = containerRef.current
                    const rect = (container ?? e.currentTarget).getBoundingClientRect()
                    const px = e.clientX - rect.left
                    const py = e.clientY - rect.top
                    const centerX = px - rect.width * 0.5
                    const centerY = py - rect.height * 0.5
                    onWheelZoom(centerX, centerY, e.deltaY)
                    break
            }
        },
        [drawMode, player, onWheelZoom]
    )

    const onMouseMove = React.useCallback(
        (e: React.MouseEvent<HTMLCanvasElement>) => {
            if (drawMode) return
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
                webPlayer.width,
                webPlayer.height,
                webPlayer.currentFrameIndex,
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
        [
            drawMode,
            player,
            zoom,
            offset.x,
            offset.y,
            hoveredTrackId,
            job.dominant_orientation,
            webPlayer.currentFrameIndex,
            webPlayer.width,
            webPlayer.height,
        ]
    )

    const onClick = React.useCallback(() => {
        if (drawMode) return
        if (!player) return

        if (player.mode === 'overview' && hoveredTrackId == null) {
            if (webPlayer.playing) {
                webPlayer.pause()
            } else if (
                webPlayer.ended ||
                (webPlayer.frameCount > 0 && webPlayer.currentFrameIndex >= webPlayer.frameCount - 1)
            ) {
                void webPlayer.seekFrame(0, true)
            } else {
                webPlayer.play()
            }
            return
        }

        if (player.mode !== 'overview' || hoveredTrackId == null) {
            setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))
            return
        }

        trackEvent('surfer_clicked', { track_id: hoveredTrackId })
        const active = player.isTrackActiveAtFrame(hoveredTrackId, webPlayer.currentFrameIndex)
        if (active) {
            if (!focusedClickHintDismissed) {
                setFocusedClickHintDismissed(true)
                void saveSetting(PLAYER_FOCUSED_CLICK_HINT_DISMISSED_KEY, true)
            }
            setPlayer(p => (p ? p.copy({ mode: 'detailed', currentTrackId: hoveredTrackId }) : p))
        } else {
            setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))
        }
    }, [
        drawMode,
        focusedClickHintDismissed,
        hoveredTrackId,
        player,
        webPlayer.currentFrameIndex,
        webPlayer.ended,
        webPlayer.frameCount,
        webPlayer.pause,
        webPlayer.play,
        webPlayer.playing,
        webPlayer.seekFrame,
    ])

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
            if (!file || !p) throw new Error(t('player.canvas.export.errors.notReady'))
            if (p.mode !== 'detailed' || p.currentTrackId == null)
                throw new Error(t('player.canvas.export.errors.selectTrack'))

            const track = p.tracks.find(t => t.track_id === p.currentTrackId)
            if (!track) throw new Error(t('player.canvas.export.errors.trackNotFound'))
            trackEvent('export_track_start', { job_id: job.id, track_id: track.track_id })

            const padSec = 0.25
            const startSec = Math.max(0, track.start_time_seconds - padSec)
            const endSec = track.start_time_seconds + track.duration_seconds + padSec
            if (!(endSec > startSec + 1e-3)) throw new Error(t('player.canvas.export.errors.trackTooShort'))

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
            const fallback = t('player.canvas.export.errors.failed')
            const message = String(e?.message || e || fallback)
            setExportError(message)
            trackEvent('export_track_failed', { job_id: job.id, message })
        } finally {
            setIsExporting(false)
            setExportProgressPct(null)
        }
    }, [isExporting, job.dominant_orientation, job.id, job.local_relative_path, player, sourceFile, t])

    const modeIndicatorLabel = React.useMemo(() => {
        if (drawMode) return t('player.canvas.modeIndicator.draw')
        if (player?.mode === 'detailed') return t('player.canvas.modeIndicator.focused')
        return t('player.canvas.modeIndicator.overview')
    }, [drawMode, player?.mode, t])

    const showFocusedClickHint = !drawMode && player?.mode === 'overview' && !focusedClickHintDismissed

    const canvasCursorClass = drawMode
        ? 'cursor-crosshair'
        : !drawMode && player?.mode === 'overview' && hoveredTrackId != null
        ? 'cursor-pointer'
        : ''

    return (
        <div className="relative flex flex-col h-full">
            <div className="relative flex-1 bg-black overflow-hidden">
                <div className="absolute right-2 top-2 z-20 pointer-events-none">
                    <div className="px-2 py-1 rounded-md bg-black/60 border border-gray-700 text-gray-100 text-[11px] font-medium">
                        {modeIndicatorLabel}
                    </div>
                </div>
                {showFocusedClickHint && (
                    <div className="absolute left-1/2 top-10 -translate-x-1/2 z-20 pointer-events-none">
                        <div className="px-3 py-2 rounded-md bg-black/70 border border-gray-700 text-gray-100 text-xs">
                            {t('player.canvas.hints.clickRiderFocused')}
                        </div>
                    </div>
                )}
                {errorText && <div className="absolute left-2 top-2 text-red-500 text-sm">{errorText}</div>}
                {webPlayer.error && (
                    <div className="absolute left-2 top-7 right-2 text-red-400 text-xs whitespace-pre-wrap break-words">
                        {webPlayer.error}
                    </div>
                )}
                {playerInitError && <div className="absolute left-2 top-7 text-red-400 text-sm">{playerInitError}</div>}
                {exportError && (
                    <div className="absolute left-2 top-7 text-red-400 text-sm">
                        {t('player.canvas.export.errorLabel', { message: exportError })}
                    </div>
                )}
                {!fileMissing && !errorText && !webPlayer.error && (webPlayer.loading || !webPlayer.ready) && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <div className="text-gray-200/80 text-sm font-mono">
                            {webPlayer.loading ? 'decoding…' : 'idle'}
                        </div>
                    </div>
                )}
                {fileMissing && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-red-500 text-3xl font-extrabold tracking-wide">
                            {t('player.canvas.fileNotFound')}
                        </div>
                    </div>
                )}
                <div ref={containerRef} className="absolute inset-0">
                    <canvas
                        ref={canvasRef}
                        className={`absolute inset-0 block ${canvasCursorClass}`}
                        onWheel={onWheelCanvas}
                        onMouseMove={onMouseMove}
                        onMouseLeave={() => setHoveredTrackId(null)}
                        onClick={onClick}
                        onPointerDown={annotations.onPointerDown}
                        onPointerMove={annotations.onPointerMove}
                        onPointerUp={annotations.onPointerUp}
                        onPointerCancel={annotations.onPointerCancel}
                    />
                    {annotations.drawModal}
                </div>
            </div>

            {player && (
                <div className="px-3 py-2 bg-black/60 border-t border-gray-700">
                    <div className="mb-2">
                        <Timeline
                            onSeekPercent={p => seekToFrame(player.frameIndexForPercent(p), false)}
                            currentProgressPercent={webPlayer.currentPercent}
                        />
                    </div>
                    <ControlsBar
                        onPlayPause={() => {
                            trackEvent('player_play_pause_clicked', { job_id: job.id })
                            handlePlayPause()
                        }}
                        onSpeedDown={() => {
                            trackEvent('player_speed_down_clicked', { job_id: job.id })
                            bumpSpeed(true)
                        }}
                        onSpeedUp={() => {
                            trackEvent('player_speed_up_clicked', { job_id: job.id })
                            bumpSpeed(false)
                        }}
                        isPlaying={webPlayer.playing}
                        speed={speed}
                        zoom={player.mode === 'detailed' ? detailedZoom.value : zoom}
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
                        <div className="text-base font-semibold">{t('player.canvas.export.overlay.title')}</div>
                        {typeof exportProgressPct === 'number' ? (
                            <div className="mt-1 text-sm tabular-nums">
                                {Math.max(0, Math.min(100, exportProgressPct)).toFixed(0)}%
                            </div>
                        ) : (
                            <div className="mt-1 text-sm">{t('player.canvas.export.overlay.starting')}</div>
                        )}
                        <div className="mt-2 text-xs text-gray-300">{t('player.canvas.export.overlay.note')}</div>
                    </div>
                </div>
            )}
        </div>
    )
}
