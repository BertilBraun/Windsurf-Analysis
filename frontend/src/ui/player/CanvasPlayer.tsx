import React from 'react'
import { useTranslation } from 'react-i18next'
import { JobDetail, ReportType } from '../types'
import { PlayerState, VideoProperties } from './state'
import { ControlsBar } from './ControlsBar'
import { Timeline } from './Timeline'
import { useZoom } from '../hooks/useZoom'
import { usePlaybackSpeed } from '../hooks/usePlaybackSpeed'
import { clamp } from '../utils/clamp'
import { trackEvent } from '../utils/analytics'
import { buildExportFilename, downloadExport, exportTrackMp4 } from './export'
import { useJobVideoSource } from './useJobVideoSource'
import { AnnotationStroke, drawFrame, pickTrackAtScreenPoint, screenPointToVideoNorm } from './rendering'
import { DrawOverlay, DRAW_COLOR_OPTIONS, DRAW_WIDTH_OPTIONS, type DrawTool } from './DrawOverlay'
import { useWebCodexPlayer, type WebCodexFrame } from './useWebCodexPlayer'

type Props = {
    job: JobDetail
    dirHandle: FileSystemDirectoryHandle | null
    onClose: () => void
    onReport: (id: string, type: ReportType, message: string) => void
    onOpenNextJob?: () => void
    onOpenPrevJob?: () => void
    drawMode: boolean
    onToggleDrawMode: () => void
}

const ANNOTATION_TIME_WINDOW_SEC = 0.1
export const CanvasPlayer: React.FC<Props> = ({
    job,
    dirHandle,
    onClose,
    onOpenNextJob,
    onOpenPrevJob,
    drawMode,
    onToggleDrawMode,
}) => {
    const { t } = useTranslation()
    const [exportError, setExportError] = React.useState<string | null>(null)
    const [playerInitError, setPlayerInitError] = React.useState<string | null>(null)
    const containerRef = React.useRef<HTMLDivElement | null>(null)
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
    const [player, setPlayer] = React.useState<PlayerState | null>(null)
    const { zoom, offset, onWheelZoom } = useZoom(containerRef)
    const { speed, bumpSpeed } = usePlaybackSpeed(1.0)
    const [hoveredTrackId, setHoveredTrackId] = React.useState<number | null>(null)
    const [isExporting, setIsExporting] = React.useState<boolean>(false)
    const [exportProgressPct, setExportProgressPct] = React.useState<number | null>(null)
    const [drawTool, setDrawTool] = React.useState<DrawTool>('line')
    const [drawColor, setDrawColor] = React.useState<string>(DRAW_COLOR_OPTIONS[0])
    const [drawWidth, setDrawWidth] = React.useState<number>(5)
    const annotationsRef = React.useRef<AnnotationStroke[]>([])
    const activeStrokeRef = React.useRef<AnnotationStroke | null>(null)
    const activePointerIdRef = React.useRef<number | null>(null)
    const [annotationsVersion, setAnnotationsVersion] = React.useState<number>(0)

    const playerRef = React.useRef<PlayerState | null>(null)
    React.useEffect(() => {
        playerRef.current = player
    }, [player])

    const lastFrameRef = React.useRef<null | {
        frameCanvas: WebCodexFrame['frameCanvas']
        width: number
        height: number
        timeSeconds: number
        frameIndex: number
        percent: number
    }>(null)

    const { sourceFile, fileMissing, error } = useJobVideoSource({
        job,
        dirHandle,
    })
    const errorText = error ? t(error.key, { message: error.detail }) : null

    const renderDecodedFrame = React.useCallback(
        (frame: WebCodexFrame) => {
            if (isExporting) return
            const fallbackW = (frame.frameCanvas as any)?.width ?? 0
            const fallbackH = (frame.frameCanvas as any)?.height ?? 0
            const frameW = frame.width || fallbackW
            const frameH = frame.height || fallbackH
            if (!frameW || !frameH) return
            lastFrameRef.current = {
                frameCanvas: frame.frameCanvas,
                width: frameW,
                height: frameH,
                timeSeconds: frame.timeSeconds,
                frameIndex: frame.frameIndex,
                percent: frame.percent,
            }

            const c = canvasRef.current
            const container = containerRef.current
            const p = playerRef.current
            if (!c || !container || !p) return

            drawFrame(
                c,
                container,
                frame.frameCanvas,
                { width: frameW, height: frameH },
                p,
                { zoom, offsetX: offset.x, offsetY: offset.y, hoveredTrackId },
                getVisibleAnnotations(frame.timeSeconds),
                frame.timeSeconds,
                job.dominant_orientation
            )

            setPlayer(prev => (prev ? prev.copy({ currentTimeSec: frame.timeSeconds }) : prev))
        },
        [isExporting, zoom, offset.x, offset.y, hoveredTrackId, job.dominant_orientation]
    )

    const webPlayer = useWebCodexPlayer({ render: renderDecodedFrame, playbackRate: speed })

    console.log('Num Frames', webPlayer.frameCount)
    console.log('Num Stabilization Transforms:', job.stabilization_transforms.length)

    const seekTo = React.useCallback(
        (timeSec: number, play: boolean) => {
            if (!webPlayer.ready) return
            const duration = playerRef.current?.video.durationSeconds ?? webPlayer.durationSeconds
            const clampedTime =
                duration && Number.isFinite(duration) ? clamp(timeSec, 0, duration) : Math.max(0, timeSec)
            setPlayer(prev => (prev ? prev.copy({ currentTimeSec: clampedTime, isPlaying: play }) : prev))
            void webPlayer.seekTimeSeconds(clampedTime, play)
        },
        [webPlayer.ready, webPlayer.durationSeconds]
    )

    React.useEffect(() => {
        trackEvent('player_open', { job_id: job.id })
        setExportError(null)
        setPlayerInitError(null)
        setIsExporting(false)
        setExportProgressPct(null)
        annotationsRef.current = []
        activeStrokeRef.current = null
        activePointerIdRef.current = null
        setAnnotationsVersion(v => v + 1)
    }, [job.id])

    React.useEffect(() => {
        let cancelled = false
        const run = async () => {
            setPlayer(null)
            lastFrameRef.current = null
            await webPlayer.dispose()
            if (cancelled) return
            if (!sourceFile) return
            await webPlayer.load(sourceFile)
        }
        void run()
        return () => {
            cancelled = true
        }
    }, [sourceFile])

    React.useEffect(() => {
        if (!webPlayer.ready) return
        try {
            const videoProps: VideoProperties = {
                width: webPlayer.width,
                height: webPlayer.height,
                durationSeconds: webPlayer.durationSeconds,
            }
            setPlayer(PlayerState.from(job, videoProps))
        } catch (e: any) {
            setPlayer(null)
            setPlayerInitError(String(e?.message ?? e ?? 'Failed to initialize player state'))
        }
    }, [job, webPlayer.ready, webPlayer.width, webPlayer.height, webPlayer.durationSeconds])

    React.useEffect(() => {
        if (!player) return
        if (!webPlayer.ready) return
        if (isExporting) {
            webPlayer.pause()
            return
        }
        if (player.isPlaying) webPlayer.play()
        else webPlayer.pause()
    }, [isExporting, player?.isPlaying, webPlayer.ready])

    // Reflect playback ending back into UI state, but don't "auto-pause" due to transient play-loop state.
    React.useEffect(() => {
        if (!player?.isPlaying) return
        if (!webPlayer.ready) return
        if (webPlayer.playing || webPlayer.loading || webPlayer.seeking) return

        const atEndByFrame = webPlayer.frameCount > 0 && webPlayer.currentFrameIndex >= webPlayer.frameCount - 1
        const atEndByTime =
            webPlayer.durationSeconds > 0 && webPlayer.currentTimeSeconds >= webPlayer.durationSeconds - 1e-3
        if (!atEndByFrame && !atEndByTime) return

        setPlayer(p => (p ? p.copy({ isPlaying: false }) : p))
    }, [
        player?.isPlaying,
        webPlayer.ready,
        webPlayer.playing,
        webPlayer.loading,
        webPlayer.seeking,
        webPlayer.currentFrameIndex,
        webPlayer.frameCount,
        webPlayer.currentTimeSeconds,
        webPlayer.durationSeconds,
    ])

    const handlePlayPause = React.useCallback(() => {
        if (drawMode) {
            onToggleDrawMode()
            setPlayer(p => (p ? p.copy({ isPlaying: true }) : p))
            return
        }
        setPlayer(p => {
            if (!p) return p
            const nextIsPlaying = !p.isPlaying
            if (nextIsPlaying && webPlayer.ready) {
                const atEndByFrame = webPlayer.frameCount > 0 && webPlayer.currentFrameIndex >= webPlayer.frameCount - 1
                const atEndByTime =
                    webPlayer.durationSeconds > 0 && webPlayer.currentTimeSeconds >= webPlayer.durationSeconds - 1e-3
                if (atEndByFrame || atEndByTime) {
                    seekTo(0, true)
                    return p.copy({ isPlaying: true, currentTimeSec: 0 })
                }
            }
            return p.copy({ isPlaying: nextIsPlaying })
        })
    }, [
        drawMode,
        onToggleDrawMode,
        webPlayer.ready,
        webPlayer.frameCount,
        webPlayer.currentFrameIndex,
        webPlayer.currentTimeSeconds,
        webPlayer.durationSeconds,
        seekTo,
    ])

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

    const getVisibleAnnotations = React.useCallback((timeSec: number) => {
        const visible = annotationsRef.current.filter(
            stroke => Math.abs(stroke.timeSec - timeSec) <= ANNOTATION_TIME_WINDOW_SEC
        )
        if (activeStrokeRef.current) visible.push(activeStrokeRef.current)
        return visible
    }, [])

    const redrawFrame = React.useCallback(
        (timeSec?: number) => {
            const c = canvasRef.current
            const container = containerRef.current
            const p = playerRef.current
            const f = lastFrameRef.current
            if (!c || !container || !p || !f) return
            if (isExporting) return
            const drawTime = timeSec ?? p.currentTimeSec
            drawFrame(
                c,
                container,
                f.frameCanvas,
                { width: f.width, height: f.height },
                p,
                { zoom, offsetX: offset.x, offsetY: offset.y, hoveredTrackId },
                getVisibleAnnotations(drawTime),
                drawTime,
                job.dominant_orientation
            )
        },
        [player, isExporting, zoom, offset.x, offset.y, hoveredTrackId, job.dominant_orientation, getVisibleAnnotations]
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
                if (webPlayer.ready && webPlayer.currentTimeSeconds >= webPlayer.durationSeconds - 0.05) {
                    trackEvent('shortcut_used', { action: 'restart_play' })
                    seekTo(0, true)
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
            } else if (e.ctrlKey && key.toLowerCase() === 'z') {
                if (!drawMode) return
                if (player?.isPlaying) return
                e.preventDefault()
                trackEvent('shortcut_used', { action: 'undo_draw' })
                if (activeStrokeRef.current) {
                    activeStrokeRef.current = null
                    activePointerIdRef.current = null
                    redrawFrame(player?.currentTimeSec)
                    return
                }
                if (annotationsRef.current.length === 0) return
                annotationsRef.current = annotationsRef.current.slice(0, -1)
                setAnnotationsVersion(v => v + 1)
                redrawFrame(player?.currentTimeSec)
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
        seekTo,
        bumpSpeed,
        goToAdjacentTrack,
        onClose,
        onOpenNextJob,
        onOpenPrevJob,
        onToggleDrawMode,
        drawMode,
        redrawFrame,
        player?.currentTimeSec,
    ])

    React.useEffect(() => {
        if (!drawMode) return
        setHoveredTrackId(null)
    }, [drawMode])

    React.useEffect(() => {
        if (!drawMode) return
        if (!player?.isPlaying) return
        setPlayer(p => (p ? p.copy({ isPlaying: false }) : p))
        webPlayer.pause()
    }, [drawMode, player?.isPlaying])

    React.useEffect(() => {
        if (!activeStrokeRef.current) return
        activeStrokeRef.current = null
        activePointerIdRef.current = null
        redrawFrame(player?.currentTimeSec)
    }, [drawTool, player?.currentTimeSec, redrawFrame])

    React.useEffect(() => {
        if (drawMode) return
        if (!activeStrokeRef.current) return
        activeStrokeRef.current = null
        activePointerIdRef.current = null
        redrawFrame(player?.currentTimeSec)
    }, [drawMode, player?.currentTimeSec, redrawFrame])

    // Redraw when paused and state changes (hover, zoom, annotations, etc).
    React.useEffect(() => {
        if (!player) return
        if (isExporting) return
        redrawFrame()
    }, [
        isExporting,
        player?.currentTimeSec,
        player?.mode,
        player?.currentTrackId,
        hoveredTrackId,
        job.dominant_orientation,
        annotationsVersion,
        redrawFrame,
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
            if (drawMode) return
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
        [drawMode, player, zoom, offset.x, offset.y, hoveredTrackId, job.dominant_orientation]
    )

    const onClick = React.useCallback(() => {
        if (drawMode) return
        const p0 = playerRef.current
        if (!p0 || hoveredTrackId == null || p0.mode !== 'overview') {
            setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))
            return
        }

        const trackId = hoveredTrackId
        trackEvent('surfer_clicked', { track_id: trackId })
        const detection = p0.interpolateDetectionByTime(trackId, p0.currentTimeSec)
        if (!detection) {
            setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))
            return
        }

        const t = detection.time_percent * p0.video.durationSeconds
        setPlayer(p => (p ? p.copy({ mode: 'detailed', currentTrackId: trackId, currentTimeSec: t }) : p))
        seekTo(t, p0.isPlaying)
    }, [drawMode, hoveredTrackId, seekTo])

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
                {
                    zoom,
                    offsetX: offset.x,
                    offsetY: offset.y,
                    hoveredTrackId,
                },
                job.dominant_orientation
            )
        },
        [player, zoom, offset.x, offset.y, hoveredTrackId, job.dominant_orientation]
    )

    const startLineStroke = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            const point = getDrawPoint(e)
            if (!point || !player) return
            activePointerIdRef.current = e.pointerId
            activeStrokeRef.current = {
                id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
                timeSec: player.currentTimeSec,
                color: drawColor,
                width: drawWidth,
                points: [point, point],
            }
            redrawFrame(player.currentTimeSec)
        },
        [getDrawPoint, player, drawColor, drawWidth, redrawFrame]
    )

    const updateLineStroke = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            if (activePointerIdRef.current !== e.pointerId) return
            const stroke = activeStrokeRef.current
            if (!stroke) return
            const point = getDrawPoint(e)
            if (!point) return
            stroke.points[1] = point
            redrawFrame(player?.currentTimeSec)
        },
        [getDrawPoint, player, redrawFrame]
    )

    const finalizeLineStroke = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            if (activePointerIdRef.current !== e.pointerId) return
            const stroke = activeStrokeRef.current
            if (!stroke) return
            const point = getDrawPoint(e)
            if (point) stroke.points[1] = point
            annotationsRef.current = [...annotationsRef.current, stroke]
            activeStrokeRef.current = null
            activePointerIdRef.current = null
            setAnnotationsVersion(v => v + 1)
            redrawFrame(player?.currentTimeSec)
        },
        [getDrawPoint, player, redrawFrame]
    )

    const onPointerDown = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            if (!drawMode || isExporting) return
            if (player?.isPlaying) return
            if (e.button !== 0) return
            e.preventDefault()
            if (drawTool === 'line') {
                if (activeStrokeRef.current) {
                    finalizeLineStroke(e)
                } else {
                    startLineStroke(e)
                }
                return
            }
            const point = getDrawPoint(e)
            if (!point || !player) return
            activePointerIdRef.current = e.pointerId
            e.currentTarget.setPointerCapture(e.pointerId)
            activeStrokeRef.current = {
                id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
                timeSec: player.currentTimeSec,
                color: drawColor,
                width: drawWidth,
                points: [point],
            }
            redrawFrame(player.currentTimeSec)
        },
        [
            drawMode,
            isExporting,
            drawTool,
            getDrawPoint,
            player,
            drawColor,
            drawWidth,
            redrawFrame,
            finalizeLineStroke,
            startLineStroke,
        ]
    )

    const onPointerMove = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            if (!drawMode || isExporting) return
            if (player?.isPlaying) return
            if (drawTool === 'line') {
                if (!activeStrokeRef.current) return
                updateLineStroke(e)
                return
            }
            if (activePointerIdRef.current !== e.pointerId) return
            const stroke = activeStrokeRef.current
            if (!stroke) return
            const point = getDrawPoint(e)
            if (!point) return
            const last = stroke.points[stroke.points.length - 1]
            const dx = point.x - last.x
            const dy = point.y - last.y
            if (dx * dx + dy * dy < 1e-7) return
            stroke.points.push(point)
            redrawFrame(player?.currentTimeSec)
        },
        [drawMode, isExporting, drawTool, getDrawPoint, player, redrawFrame, updateLineStroke]
    )

    const finalizeStroke = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>, canceled: boolean) => {
            if (!drawMode) return
            if (player?.isPlaying) return
            if (drawTool !== 'freehand') return
            if (activePointerIdRef.current !== e.pointerId) return
            if (e.currentTarget.hasPointerCapture(e.pointerId)) {
                e.currentTarget.releasePointerCapture(e.pointerId)
            }
            activePointerIdRef.current = null
            const stroke = activeStrokeRef.current
            activeStrokeRef.current = null
            if (stroke && !canceled) {
                annotationsRef.current = [...annotationsRef.current, stroke]
                setAnnotationsVersion(v => v + 1)
            }
            redrawFrame(player?.currentTimeSec)
        },
        [drawMode, drawTool, player, redrawFrame]
    )

    const cancelLineStroke = React.useCallback(
        (e: React.PointerEvent<HTMLCanvasElement>) => {
            if (drawTool !== 'line') return
            if (player?.isPlaying) return
            if (activePointerIdRef.current !== e.pointerId) return
            if (!activeStrokeRef.current) return
            activeStrokeRef.current = null
            activePointerIdRef.current = null
            redrawFrame(player?.currentTimeSec)
        },
        [drawTool, player, redrawFrame]
    )

    const onClearAnnotations = React.useCallback(() => {
        if (!player) return
        const nowSec = player.currentTimeSec
        annotationsRef.current = annotationsRef.current.filter(
            stroke => Math.abs(stroke.timeSec - nowSec) > ANNOTATION_TIME_WINDOW_SEC
        )
        activeStrokeRef.current = null
        activePointerIdRef.current = null
        setAnnotationsVersion(v => v + 1)
        redrawFrame(nowSec)
    }, [player, redrawFrame])

    const hasVisibleAnnotations = React.useMemo(() => {
        if (!player) return false
        const nowSec = player.currentTimeSec
        return annotationsRef.current.some(stroke => Math.abs(stroke.timeSec - nowSec) <= ANNOTATION_TIME_WINDOW_SEC)
    }, [player?.currentTimeSec, annotationsVersion])

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
            const endSec = Math.min(
                p.video.durationSeconds || Infinity,
                track.start_time_seconds + track.duration_seconds + padSec
            )
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

    return (
        <div className="relative flex flex-col h-full">
            <div className="relative flex-1 bg-black overflow-hidden">
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
                        className={`absolute inset-0 block ${drawMode ? 'cursor-crosshair' : ''}`}
                        onWheel={onWheelCanvas}
                        onMouseMove={onMouseMove}
                        onMouseLeave={() => setHoveredTrackId(null)}
                        onClick={onClick}
                        onPointerDown={onPointerDown}
                        onPointerMove={onPointerMove}
                        onPointerUp={e => finalizeStroke(e, false)}
                        onPointerCancel={e => {
                            finalizeStroke(e, true)
                            cancelLineStroke(e)
                        }}
                    />
                    {drawMode && (
                        <DrawOverlay
                            drawTool={drawTool}
                            onDrawToolChange={setDrawTool}
                            drawWidth={drawWidth}
                            onDrawWidthChange={setDrawWidth}
                            drawColor={drawColor}
                            onDrawColorChange={setDrawColor}
                            onClearAnnotations={onClearAnnotations}
                            hasVisibleAnnotations={hasVisibleAnnotations}
                        />
                    )}
                </div>
            </div>

            {player && (
                <div className="px-3 py-2 bg-black/60 border-t border-gray-700">
                    <div className="mb-2">
                        <Timeline
                            onSeekTime={t => seekTo(t, false)}
                            percent01={webPlayer.currentPercent}
                            duration={webPlayer.durationSeconds}
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
