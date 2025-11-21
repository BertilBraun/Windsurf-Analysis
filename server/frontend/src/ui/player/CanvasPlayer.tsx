import React from 'react'
import { JobDetail, ReportType } from '../types'
import { getFileByRelativePath } from '../utils/fsAccess'
import { PlayerState, VideoProperties } from './state'
import { ControlsBar } from './ControlsBar'
import { Timeline } from './Timeline'
import { TARGET_BBOX_HEIGHT_RATIO, MIN_SCALE, MAX_SCALE } from './constants'
import { useZoom } from '../hooks/useZoom'
import { usePlaybackSpeed } from '../hooks/usePlaybackSpeed'
import { useSeeker } from '../hooks/useSeeker'
import { clamp } from '../utils/clamp'
import { drawRotatedToCanvas, getRotatedDimensions } from './rotation'

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

export const CanvasPlayer: React.FC<Props> = ({ job, dirHandle, onClose, onOpenNextJob, onOpenPrevJob }) => {
    const [error, setError] = React.useState<string | null>(null)
    const [fileMissing, setFileMissing] = React.useState<boolean>(false)
    const [videoUrl, setVideoUrl] = React.useState<string | null>(null)
    const videoRef = React.useRef<HTMLVideoElement | null>(null)
    const containerRef = React.useRef<HTMLDivElement | null>(null)
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null)
    const [player, setPlayer] = React.useState<PlayerState | null>(null)
    const { zoom, offset, onWheelZoom } = useZoom(containerRef)
    const { speed, bumpSpeed } = usePlaybackSpeed(1.0)
    const [hoveredTrackId, setHoveredTrackId] = React.useState<number | null>(null)

    // Resolve file URL
    const resolveFileFromRelativePath = React.useCallback(async () => {
        if (!dirHandle) {
            setError('No ingress folder selected')
            console.log('No dirHandle; cannot resolve', job.local_relative_path)
            return null
        }
        try {
            const path = job.local_relative_path
            if (!path) throw new Error('Missing local mapping for file')
            return await getFileByRelativePath(dirHandle, path)
        } catch (e: any) {
            const msg = String(e?.message || '')
            const isMissing = e?.name === 'NotFoundError' || /not\s*found|no such file|could not be found/i.test(msg)
            if (isMissing) {
                setFileMissing(true)
                setError('VIDEO FILE NOT FOUND')
            } else {
                setError(msg || 'Failed to access file from folder')
            }
            console.log('Error resolving file', e)
            return null
        }
    }, [dirHandle, job.local_relative_path])

    React.useEffect(() => {
        let revoked: string | null = null
        setVideoUrl(null)
        setError(null)
        setFileMissing(false)
        resolveFileFromRelativePath().then(file => {
            if (!file) return
            const url = URL.createObjectURL(file)
            revoked = url
            setVideoUrl(url)
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
        v.defaultPlaybackRate = speed
        v.playbackRate = speed
        if (player.isPlaying) v.play().catch(() => {})
        else v.pause()
    }, [player?.isPlaying, speed])

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
            if (!player) return
            const key = e.key.length === 1 ? e.key.toLowerCase() : e.key
            if (key === ' ') {
                e.preventDefault()
                const v = videoRef.current
                if (v && v.duration && v.currentTime >= v.duration - 0.05) {
                    seekTo(0, true)
                } else {
                    togglePlay()
                }
            } else if (key === 'ArrowLeft' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault()
                stepPrev()
            } else if (key === 'ArrowRight' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault()
                stepNext()
            } else if (e.ctrlKey && key === 'ArrowLeft') {
                e.preventDefault()
                seekTo(player.currentTimeSec - 30, true)
            } else if (e.ctrlKey && key === 'ArrowRight') {
                e.preventDefault()
                seekTo(player.currentTimeSec + 30, true)
            } else if (e.shiftKey && key === 'ArrowLeft') {
                e.preventDefault()
                seekTo(player.currentTimeSec - 5, true)
            } else if (e.shiftKey && key === 'ArrowRight') {
                e.preventDefault()
                seekTo(player.currentTimeSec + 5, true)
            } else if (key === '-') {
                e.preventDefault()
                bumpSpeed(true)
            } else if (key === '+' || key === '=') {
                e.preventDefault()
                bumpSpeed(false)
            } else if (key.toLowerCase() === 'escape') {
                if (player.mode === 'overview') {
                    onClose?.()
                } else {
                    setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))
                }
            } else if (!e.shiftKey && key.toLowerCase() === 'n') {
                e.preventDefault()
                goToAdjacentTrack(true)
            } else if (!e.shiftKey && key.toLowerCase() === 'p') {
                e.preventDefault()
                goToAdjacentTrack(false)
            } else if (e.shiftKey && key.toLowerCase() === 'n') {
                e.preventDefault()
                onOpenNextJob?.()
            } else if (e.shiftKey && key.toLowerCase() === 'p') {
                e.preventDefault()
                onOpenPrevJob?.()
            }
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [
        player,
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
    }, [player?.isPlaying, player?.mode, player?.currentTrackId, zoom, offset.x, offset.y, hoveredTrackId, videoUrl])

    // Redraw on resize or when paused and state changes
    React.useEffect(() => {
        const c = canvasRef.current
        const v = videoRef.current
        const container = containerRef.current
        if (!c || !v || !player || !container) return
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
        if (!player || player.mode !== 'detailed' || player.currentTrackId == null) return
        if (!player.hasDetectionAfter(player.currentTrackId, player.currentTimeSec)) {
            setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))
        }
    }, [player?.currentTimeSec, player?.mode, player?.currentTrackId])

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
            const detection = p.interpolateDetectionByTime(trackId, p.currentTimeSec)
            if (!detection) return p.copy({ mode: 'overview', currentTrackId: null })

            const t = detection.time_percent * p.video.durationSeconds
            videoRef.current!.currentTime = t
            return p.copy({ mode: 'detailed', currentTrackId: trackId, currentTimeSec: t })
        })
    }, [hoveredTrackId])

    return (
        <div className="flex flex-col h-full">
            <div className="relative flex-1 bg-black overflow-hidden">
                {error && <div className="absolute left-2 top-2 text-red-500 text-sm">{error}</div>}
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
                        onPlayPause={togglePlay}
                        onSpeedDown={() => bumpSpeed(true)}
                        onSpeedUp={() => bumpSpeed(false)}
                        isPlaying={player.isPlaying}
                        speed={speed}
                        zoom={zoom}
                    />
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
            const det = player.interpolateDetectionByTime(t.track_id, now)
            if (!det) continue
            const [x1p, y1p, x2p, y2p] = det.bbox
            const x1 = x1p * rotatedVideo.width - rotatedVideo.width * 0.5
            const y1 = y1p * rotatedVideo.height - rotatedVideo.height * 0.5
            const w = Math.max(1, (x2p - x1p) * rotatedVideo.width)
            const h = Math.max(1, (y2p - y1p) * rotatedVideo.height)
            const isHovered = ov.hoveredTrackId === t.track_id
            ctx.strokeStyle = isHovered ? '#10b981' : '#ef4444'
            ctx.lineWidth = 2 / sBase
            ctx.strokeRect(Math.round(x1) + 0.5, Math.round(y1) + 0.5, Math.round(w), Math.round(h))

            drawText(ctx, String(t.track_id), x1, y1, '#fff', 'rgba(0,0,0,0.7)')
        }
        ctx.restore()
    } else if (player.mode === 'detailed' && player.currentTrackId != null) {
        const now = timeOverrideSec ?? player.currentTimeSec
        const det = player.interpolateDetectionByTime(player.currentTrackId, now)
        if (!det) return

        const vidW = rotatedVideo.width
        const vidH = rotatedVideo.height
        const [x1p, y1p, x2p, y2p] = det.bbox
        const x1 = x1p * vidW
        const y1 = y1p * vidH
        const x2 = x2p * vidW
        const y2 = y2p * vidH
        const bboxW = Math.max(1, x2 - x1)
        const bboxH = Math.max(1, y2 - y1)

        const sHeight = (TARGET_BBOX_HEIGHT_RATIO * cssH) / bboxH
        const sWidthLimit = cssW / bboxW
        const s = clamp(Math.min(sHeight, sWidthLimit), MIN_SCALE, MAX_SCALE)

        const cx = (x1 + x2) * 0.5
        const cy = (y1 + y2) * 0.5
        const cropW = cssW / s
        const cropH = cssH / s
        const winX1 = cx - cropW / 2
        const winY1 = cy - cropH / 2
        const winX2 = winX1 + cropW
        const winY2 = winY1 + cropH

        const srcX1 = Math.max(0, Math.floor(winX1))
        const srcY1 = Math.max(0, Math.floor(winY1))
        const srcX2 = Math.min(vidW, Math.ceil(winX2))
        const srcY2 = Math.min(vidH, Math.ceil(winY2))
        const dstX1 = Math.max(0, Math.floor((srcX1 - winX1) * s))
        const dstY1 = Math.max(0, Math.floor((srcY1 - winY1) * s))
        const dstX2 = Math.min(cssW, Math.ceil((srcX2 - winX1) * s))
        const dstY2 = Math.min(cssH, Math.ceil((srcY2 - winY1) * s))
        const srcW = clamp(srcX2 - srcX1, 0, vidW)
        const srcH = clamp(srcY2 - srcY1, 0, vidH)
        const dstW = clamp(dstX2 - dstX1, 0, cssW)
        const dstH = clamp(dstY2 - dstY1, 0, cssH)

        if (srcW > 0 && srcH > 0 && dstW > 0 && dstH > 0) {
            try {
                ctx.imageSmoothingEnabled = true
                ctx.imageSmoothingQuality = 'high'
                ctx.drawImage(sourceCanvas, srcX1, srcY1, srcW, srcH, dstX1, dstY1, dstW, dstH)
            } catch {}
        }

        // optional: draw a subtle bbox overlay for context
        ctx.strokeStyle = '#f59e0b'
        ctx.lineWidth = 2
        const bboxScreenX = (x1 - winX1) * s
        const bboxScreenY = (y1 - winY1) * s
        const bboxScreenW = bboxW * s
        const bboxScreenH = bboxH * s
        ctx.strokeRect(
            Math.round(bboxScreenX) + 0.5,
            Math.round(bboxScreenY) + 0.5,
            Math.round(bboxScreenW),
            Math.round(bboxScreenH)
        )
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
    // apply stabilization transform to the base rect
    // TODO also apply the rotation transform
    const dx = base.x + stab.dx + ov.offsetX
    const dy = base.y + stab.dy + ov.offsetY
    const dw = base.w * ov.zoom
    const dh = base.h * ov.zoom

    // check from topmost to bottommost; here just iterate, but prefer smaller boxes first
    for (const t of player.tracks) {
        const det = player.interpolateDetectionByTime(t.track_id, player.currentTimeSec)
        if (!det) continue
        const [x1p, y1p, x2p, y2p] = det.bbox
        const x1 = dx + x1p * dw
        const y1 = dy + y1p * dh
        const x2 = dx + x2p * dw
        const y2 = dy + y2p * dh
        if (px >= x1 && px <= x2 && py >= y1 && py <= y2) return t.track_id
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
