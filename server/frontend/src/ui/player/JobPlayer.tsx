import React from 'react'
import { JobDetail, ReportType } from '../types'
import { getFileByRelativePath } from '../utils/fsAccess'
import { buildPlayerState, PlayerState, VideoProperties, findNearestDetectionByTime } from './state'
import { ControlsBar } from './ControlsBar'
import { Timeline } from './Timeline'
import { VideoOverlay } from './VideoOverlay'
import { DetailedCanvas } from './DetailedCanvas'

export const JobPlayer: React.FC<{
    job: JobDetail
    dirHandle: FileSystemDirectoryHandle | null
    onClose: () => void
    onDelete: (id: string) => void
    onReport: (id: string, type: ReportType, message: string) => void
}> = ({ job, dirHandle, onClose }) => {
    const [videoUrl, setVideoUrl] = React.useState<string | null>(null)
    const [error, setError] = React.useState<string | null>(null)
    const videoRef = React.useRef<HTMLVideoElement | null>(null)
    const containerRef = React.useRef<HTMLDivElement | null>(null)
    const [player, setPlayer] = React.useState<PlayerState | null>(null)
    const [overviewZoom, setOverviewZoom] = React.useState(1)
    const [overviewOffset, setOverviewOffset] = React.useState<{ x: number; y: number }>({ x: 0, y: 0 })

    const log = React.useCallback((...args: any[]) => {
        // Prefix logs for easy filtering
        // eslint-disable-next-line no-console
        console.log('[JobPlayer]', ...args)
    }, [])

    const resolveFileFromRelativePath = React.useCallback(async () => {
        if (!dirHandle) {
            setError('No ingress folder selected')
            log('No dirHandle; cannot resolve', job.original_file_path)
            return null
        }
        try {
            const file = await getFileByRelativePath(dirHandle, job.original_file_path)
            log('Resolved file', { name: file.name, size: file.size, type: file.type })
            return file
        } catch (e: any) {
            setError(e?.message || 'Failed to access file from folder')
            log('Error resolving file', e)
            return null
        }
    }, [dirHandle, job.original_file_path, log])

    React.useEffect(() => {
        let revoked: string | null = null
        setVideoUrl(null)
        setError(null)
        ;(async () => {
            log('Begin resolve for job', job.id)
            const file = await resolveFileFromRelativePath()
            if (!file) return
            const url = URL.createObjectURL(file)
            log('Created object URL', url)
            revoked = url
            setVideoUrl(url)
        })()
        return () => {
            if (revoked) {
                log('Revoking object URL', revoked)
                URL.revokeObjectURL(revoked)
            }
        }
    }, [resolveFileFromRelativePath, job.id, log])

    React.useEffect(() => {
        const v = videoRef.current
        if (!v) return
        const onLoadedMetadata = () => {
            const videoProps: VideoProperties = {
                width: v.videoWidth,
                height: v.videoHeight,
                durationSeconds: v.duration,
            }
            log('video loadedmetadata', videoProps)
            setPlayer(buildPlayerState(job, videoProps))
        }
        const onError = () => log('video error', v.error)
        v.addEventListener('loadedmetadata', onLoadedMetadata)
        v.addEventListener('error', onError)
        return () => {
            v.removeEventListener('loadedmetadata', onLoadedMetadata)
            v.removeEventListener('error', onError)
        }
    }, [videoUrl, log, job])

    // Sync currentTimeSec to video's currentTime; control playback via video APIs
    React.useEffect(() => {
        let raf = 0
        const v = videoRef.current
        if (!player || !v) return
        const loop = () => {
            const t = v.currentTime
            if (player && t !== player.currentTimeSec) setPlayer(prev => (prev ? { ...prev, currentTimeSec: t } : prev))
            raf = requestAnimationFrame(loop)
        }
        raf = requestAnimationFrame(loop)
        return () => cancelAnimationFrame(raf)
    }, [videoRef.current, player?.currentTimeSec])

    React.useEffect(() => {
        const v = videoRef.current
        if (!player || !v) return
        v.playbackRate = player.playbackSpeed
        if (player.isPlaying) v.play().catch(() => {})
        else v.pause()
    }, [player?.isPlaying, player?.playbackSpeed])

    const togglePlay = () => setPlayer(p => (p ? { ...p, isPlaying: !p.isPlaying } : p))

    const bumpSpeed = (down: boolean) =>
        setPlayer(p => {
            if (!p) return p
            const rates = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
            const idx = Math.max(
                0,
                Math.min(rates.length - 1, Math.max(0, rates.indexOf(p.playbackSpeed)) + (down ? -1 : 1))
            )
            return { ...p, playbackSpeed: rates[idx] }
        })

    const seekTime = (timeSec: number) => {
        const v = videoRef.current
        if (!player || !v) return
        const t = Math.max(0, Math.min(v.duration || 0, timeSec))
        v.currentTime = t
        setPlayer(p => (p ? { ...p, isPlaying: false, currentTimeSec: t } : p))
    }
    const stepNext = () => {
        const v = videoRef.current
        if (!player || !v) return
        v.currentTime = Math.min(v.duration || 0, player.currentTimeSec + 1 / 25)
    }
    const stepPrev = () => {
        const v = videoRef.current
        if (!player || !v) return
        v.currentTime = Math.max(0, player.currentTimeSec - 1 / 25)
    }
    const enterDetailed = (trackId: number) => {
        setPlayer(p => {
            if (!p) return p
            const arr = p.detectionTimesByTrack.get(trackId) || []
            const nearest = findNearestDetectionByTime(arr, p.currentTimeSec, 0.2)
            if (nearest) {
                // Seek to the detection time so detailed mode persists
                const v = videoRef.current
                if (v) v.currentTime = nearest.timeSec
                return { ...p, currentTrackId: trackId, mode: 'detailed', currentTimeSec: nearest.timeSec }
            }
            return { ...p, currentTrackId: trackId, mode: 'detailed' }
        })
    }
    const exitDetailed = () => setPlayer(p => (p ? { ...p, mode: 'overview', currentTrackId: null } : p))

    const onWheelZoom = (absX: number, absY: number, deltaY: number) => {
        const rect = containerRef.current?.getBoundingClientRect()
        const cx = rect ? absX - rect.left : absX
        const cy = rect ? absY - rect.top : absY
        let nz = overviewZoom * (1 + (deltaY < 0 ? 0.1 : -0.1))
        if (nz <= 1) {
            nz = 1
            setOverviewZoom(1)
            setOverviewOffset({ x: 0, y: 0 })
            return
        }
        const scaleChange = nz / overviewZoom
        const nx = cx - scaleChange * (cx - overviewOffset.x)
        const ny = cy - scaleChange * (cy - overviewOffset.y)
        setOverviewZoom(nz)
        setOverviewOffset({ x: nx, y: ny })
    }

    // Keyboard controls mirroring Python keyPressEvent
    React.useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (!player) return
            if (e.key === ' ') {
                e.preventDefault()
                togglePlay()
                return
            }
            if (e.key === 'ArrowLeft' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault()
                stepPrev()
                return
            }
            if (e.key === 'ArrowRight' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault()
                stepNext()
                return
            }
            if (e.ctrlKey && e.key === 'ArrowLeft') {
                e.preventDefault()
                const v = videoRef.current
                if (!v) return
                v.currentTime = Math.max(0, player.currentTimeSec - 30)
                return
            }
            if (e.ctrlKey && e.key === 'ArrowRight') {
                e.preventDefault()
                const v = videoRef.current
                if (!v) return
                v.currentTime = Math.min(v.duration || 0, player.currentTimeSec + 30)
                return
            }
            if (e.shiftKey && e.key === 'ArrowLeft') {
                e.preventDefault()
                const v = videoRef.current
                if (!v) return
                v.currentTime = Math.max(0, player.currentTimeSec - 5)
                return
            }
            if (e.shiftKey && e.key === 'ArrowRight') {
                e.preventDefault()
                const v = videoRef.current
                if (!v) return
                v.currentTime = Math.min(v.duration || 0, player.currentTimeSec + 5)
                return
            }
            if (e.key === '-') bumpSpeed(true)
            else if (e.key === '+' || e.key === '=') bumpSpeed(false)
            else if (e.key.toLowerCase() === 'escape') exitDetailed()
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [player])

    // Auto-exit detailed mode if no nearby detection at current time
    React.useEffect(() => {
        if (!player || player.mode !== 'detailed' || player.currentTrackId == null) return
        const arr = player.detectionTimesByTrack.get(player.currentTrackId) || []
        const nearest = findNearestDetectionByTime(arr, player.currentTimeSec, 0.2)
        if (!nearest) setPlayer(p => (p ? { ...p, mode: 'overview', currentTrackId: null } : p))
    }, [player?.currentTimeSec, player?.mode, player?.currentTrackId])

    return (
        <div className="flex flex-col h-full">
            {/* Header bar */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700 bg-black/60">
                <div className="text-sm text-gray-200 truncate" title={job.original_file_path}>
                    {job.original_file_path || '(unknown)'}
                </div>
                <div className="flex gap-2 items-center">
                    <button onClick={onClose}>Close</button>
                </div>
            </div>

            {/* Video area */}
            <div className="relative flex-1 bg-black overflow-hidden">
                {error && <div className="absolute left-2 top-2 text-red-500 text-sm">{error}</div>}
                {videoUrl ? (
                    <div ref={containerRef} className="absolute inset-0">
                        <div
                            className="absolute inset-0"
                            style={{
                                transform:
                                    player?.mode === 'overview'
                                        ? `translate(${overviewOffset.x}px, ${overviewOffset.y}px) scale(${overviewZoom})`
                                        : 'none',
                                transformOrigin: '0 0',
                            }}
                        >
                            <video
                                ref={videoRef}
                                key={videoUrl}
                                src={videoUrl}
                                playsInline
                                muted={false}
                                preload="metadata"
                                className="w-full h-full object-contain"
                            />
                            {player && player.mode === 'overview' && (
                                <VideoOverlay
                                    state={player}
                                    videoRef={videoRef}
                                    onEnterDetailed={enterDetailed}
                                    onWheelZoom={onWheelZoom}
                                />
                            )}
                        </div>
                        {player && player.mode === 'detailed' && <DetailedCanvas state={player} videoRef={videoRef} />}
                    </div>
                ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500">
                        Loading video…
                    </div>
                )}
            </div>

            {/* Bottom controls */}
            {player && (
                <div className="px-3 py-2 bg-black/60 border-t border-gray-700">
                    <div className="mb-2">
                        <Timeline state={player} onSeekTime={seekTime} />
                    </div>
                    <ControlsBar
                        onPlayPause={togglePlay}
                        onSpeedDown={() => bumpSpeed(true)}
                        onSpeedUp={() => bumpSpeed(false)}
                        isPlaying={player.isPlaying}
                        speed={player.playbackSpeed}
                        zoom={overviewZoom}
                    />
                </div>
            )}
        </div>
    )
}
