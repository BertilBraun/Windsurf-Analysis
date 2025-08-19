import React from 'react'
import { JobDetail, ReportType } from '../types'
import { getFileByRelativePath } from '../utils/fsAccess'
import { PlayerState, VideoProperties } from './state'
import { ControlsBar } from './ControlsBar'
import { Timeline } from './Timeline'
import { VideoOverlay } from './VideoOverlay'
import { DetailedCanvas } from './DetailedCanvas'
import { useZoom } from './useZoom'
import { usePlaybackSpeed } from './usePlaybackSpeed'
import { useSeeker } from './useSeeker'

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
    const { zoom: overviewZoom, offset: overviewOffset, onWheelZoom } = useZoom(containerRef)
    const { speed, bumpSpeed } = usePlaybackSpeed(1.0)
    const { seekTo, stepNext, stepPrev, onNewFile } = useSeeker(videoRef, player, setPlayer)

    const resolveFileFromRelativePath = React.useCallback(async () => {
        if (!dirHandle) {
            setError('No ingress folder selected')
            console.log('No dirHandle; cannot resolve', job.original_file_path)
            return null
        }
        try {
            const file = await getFileByRelativePath(dirHandle, job.original_file_path)
            console.log('Resolved file', { name: file.name, size: file.size, type: file.type })
            return file
        } catch (e: any) {
            setError(e?.message || 'Failed to access file from folder')
            console.log('Error resolving file', e)
            return null
        }
    }, [dirHandle, job.original_file_path])

    React.useEffect(() => {
        let revoked: string | null = null
        setVideoUrl(null)
        setError(null)
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

    React.useEffect(() => {
        const v = videoRef.current
        if (!v) return
        const onLoadedMetadata = () => {
            const videoProps: VideoProperties = {
                width: v.videoWidth,
                height: v.videoHeight,
                durationSeconds: v.duration,
            }
            console.log('video loadedmetadata', videoProps)
            setPlayer(PlayerState.from(job, videoProps))
        }
        const onError = () => console.log('video error', v.error)
        v.addEventListener('loadedmetadata', onLoadedMetadata)
        v.addEventListener('error', onError)
        return () => {
            v.removeEventListener('loadedmetadata', onLoadedMetadata)
            v.removeEventListener('error', onError)
        }
    }, [videoUrl, job])

    // Sync currentTimeSec precisely to decoded frames using requestVideoFrameCallback
    React.useEffect(() => {
        const v = videoRef.current
        if (!player || !v) return
        let vfId: number | null = null

        const onFrame = () => {
            setPlayer(prev => (prev ? prev.copy({ currentTimeSec: v.currentTime }) : prev))
            vfId = v.requestVideoFrameCallback(onFrame)
        }
        vfId = v.requestVideoFrameCallback(onFrame)
        return () => {
            if (vfId) v.cancelVideoFrameCallback(vfId)
        }
    }, [videoUrl, !!player])

    React.useEffect(() => {
        const v = videoRef.current
        if (!player || !v) return
        v.playbackRate = speed
        if (player.isPlaying) v.play().catch(() => {})
        else v.pause()
    }, [player?.isPlaying, speed])

    const togglePlay = () => setPlayer(p => (p ? p.togglePlay() : p))

    const enterDetailed = (trackId: number) => {
        setPlayer(p => {
            if (!p) return p
            const nearest = p.interpolateDetectionByTime(trackId, p.currentTimeSec)
            if (nearest) {
                // Seek to the detection time so detailed mode persists
                const t = nearest.time_percent * p.video.durationSeconds
                const v = videoRef.current
                if (v) v.currentTime = t
                return p.copy({ mode: 'detailed', currentTrackId: trackId, currentTimeSec: t })
            }
            return p.copy({ mode: 'detailed', currentTrackId: trackId })
        })
    }
    const exitDetailed = () => setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))

    React.useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (!player) return

            if (e.key === ' ') {
                e.preventDefault()
                togglePlay()
            } else if (e.key === 'ArrowLeft' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault()
                stepPrev()
            } else if (e.key === 'ArrowRight' && !e.ctrlKey && !e.shiftKey) {
                e.preventDefault()
                stepNext()
            } else if (e.ctrlKey && e.key === 'ArrowLeft') {
                e.preventDefault()
                seekTo(player.currentTimeSec - 30, true)
            } else if (e.ctrlKey && e.key === 'ArrowRight') {
                e.preventDefault()
                seekTo(player.currentTimeSec + 30, true)
            } else if (e.shiftKey && e.key === 'ArrowLeft') {
                e.preventDefault()
                seekTo(player.currentTimeSec - 5, true)
            } else if (e.shiftKey && e.key === 'ArrowRight') {
                e.preventDefault()
                seekTo(player.currentTimeSec + 5, true)
            } else if (e.key === '-') bumpSpeed(true)
            else if (e.key === '+' || e.key === '=') bumpSpeed(false)
            else if (e.key.toLowerCase() === 'escape') exitDetailed()
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [player])

    // Auto-exit detailed mode if no reasonably recent detection around current time
    React.useEffect(() => {
        if (!player || player.mode !== 'detailed' || player.currentTrackId == null) return

        if (!player.hasDetectionAfter(player.currentTrackId, player.currentTimeSec)) {
            setPlayer(p => (p ? p.copy({ mode: 'overview', currentTrackId: null }) : p))
        }
    }, [player?.currentTimeSec, player?.mode, player?.currentTrackId])

    return (
        <div className="flex flex-col h-full">
            {/* Header bar */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700 bg-black/60">
                <div className="text-sm text-gray-200 truncate" title={job.original_file_path}>
                    {job.original_file_path}
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
                        {player && player.mode === 'detailed' && player.currentTrackId != null && (
                            <DetailedCanvas state={player} videoRef={videoRef} />
                        )}
                    </div>
                ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-500">
                        Loading video…
                    </div>
                )}
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
                        zoom={overviewZoom}
                    />
                </div>
            )}
        </div>
    )
}
