import React from 'react'
import { clamp } from '../utils/clamp'
import { useVideoFps } from './useVideoFps'
import type { PlayerState } from './state'

export function useSeeker(
    videoRef: React.RefObject<HTMLVideoElement>,
    player: PlayerState | null,
    setPlayer: React.Dispatch<React.SetStateAction<PlayerState | null>>
) {
    const { frameDuration, analyzeFile, reset } = useVideoFps()

    const seekTo = React.useCallback(
        (timeSec: number, play: boolean) => {
            const v = videoRef.current
            if (!player || !v) return
            const t = clamp(timeSec, 0, v.duration)
            v.currentTime = t
            setPlayer(p => (p ? p.copy({ currentTimeSec: t, isPlaying: play }) : p))
        },
        [player, videoRef, setPlayer]
    )

    const stepNext = React.useCallback(() => {
        const v = videoRef.current
        if (!player || !v || !frameDuration) return
        seekTo((player?.currentTimeSec || 0) + frameDuration, false)
    }, [player, videoRef, frameDuration, seekTo])

    const stepPrev = React.useCallback(() => {
        const v = videoRef.current
        if (!player || !v || !frameDuration) return
        seekTo((player?.currentTimeSec || 0) - frameDuration, false)
    }, [player, videoRef, frameDuration, seekTo])

    const onNewFile = React.useCallback(
        (file: File) => {
            reset()
            analyzeFile(file)
        },
        [reset, analyzeFile]
    )

    return { seekTo, stepNext, stepPrev, onNewFile }
}
