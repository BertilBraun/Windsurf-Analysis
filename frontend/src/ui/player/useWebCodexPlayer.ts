/**
 * @module useWebCodexPlayer
 * Provides a React hook for high-performance, frame-accurate video playback using the WebCodecs API.
 * This module handles video demuxing, decoding, and frame caching for smooth seeking and playback.
 */

import React from 'react'
import { ALL_FORMATS, BlobSource, CanvasSink, Input, type WrappedCanvas } from 'mediabunny'
import { getSortedVideoPacketMeta } from '../../media/videoPackets'
import { clamp } from '../utils/clamp'

/**
 * Represents a single decoded frame and its associated metadata.
 */
export type WebCodexFrame = {
    /** The zero-based index of the frame in the video sequence. */
    frameIndex: number
    /** The temporal position of the frame as a percentage (0 to 1). */
    percent: number
    /** The intrinsic width of the decoded frame. */
    width: number
    /** The intrinsic height of the decoded frame. */
    height: number
    /** The canvas element or offscreen canvas containing the frame's pixel data. */
    frameCanvas: HTMLCanvasElement | OffscreenCanvas
}

/**
 * The public API and state for controlling the WebCodecs-based video player.
 */
export type WebCodexPlayerApi = {
    /** Whether the player is initialized and ready for playback. */
    ready: boolean
    /** Whether a file is currently being loaded and processed. */
    loading: boolean
    /** Whether a seek operation is currently in progress. */
    seeking: boolean
    /** Whether the video is currently playing. */
    playing: boolean
    /** Whether the video has reached the last frame. */
    ended: boolean
    /** Contains an error message if a player operation failed. */
    error?: string

    /** Total number of frames in the loaded video. */
    frameCount: number
    /** The index of the currently displayed frame. */
    currentFrameIndex: number
    /** The current playback position as a percentage (0 to 1). */
    currentPercent: number
    /** The canvas containing the current frame's image data. Null if no frame is loaded. */
    currentFrameCanvas: HTMLCanvasElement | OffscreenCanvas | null

    /** The display width of the video in pixels. */
    width: number
    /** The display height of the video in pixels. */
    height: number

    /**
     * Loads a video file into the player.
     * @param file The video file to load.
     */
    load: (file: File) => Promise<void>
    /** Releases all resources, stops playback, and resets the player state. */
    dispose: () => Promise<void>

    /** Starts video playback. */
    play: () => void
    /** Pauses video playback. */
    pause: () => void
    /** Toggles between play and pause states. */
    togglePlay: () => void

    /**
     * Seeks to a specific percentage of the video.
     * @param p Percentage from 0 to 1.
     * @param playAfter Whether to resume playback after seeking. Defaults to current playing state.
     */
    seekPercent: (p: number, playAfter?: boolean) => Promise<void>
    /**
     * Seeks to a specific frame index.
     * @param i The target frame index.
     * @param playAfter Whether to resume playback after seeking. Defaults to current playing state.
     */
    seekFrame: (i: number, playAfter?: boolean) => Promise<void>

    /**
     * Moves the playback position by a specific number of frames.
     * @param delta Number of frames to move (positive for forward, negative for backward).
     */
    stepFrames: (delta: number) => Promise<void>
}

function sleep(ms: number) {
    return new Promise<void>(r => setTimeout(r, ms))
}

/**
 * A React hook that provides a frame-accurate video player powered by WebCodecs.
 *
 * @param params Configuration for the player.
 * @param params.behindSeconds Number of seconds to keep in the decode cache behind the current position.
 * @param params.aheadSeconds Number of seconds to keep in the decode cache ahead of the current position.
 * @param params.playbackRate The speed at which the video should play (e.g., 1.0 for normal speed).
 * @returns An object containing the player state and control methods.
 */
export function useWebCodexPlayer(params: {
    behindSeconds?: number
    aheadSeconds?: number
    playbackRate?: number
}): WebCodexPlayerApi {
    const { behindSeconds = 0.5, aheadSeconds = 0.5, playbackRate = 1.0 } = params

    const [loading, setLoading] = React.useState(false)
    const [seeking, setSeeking] = React.useState(false)
    const [ready, setReady] = React.useState(false)
    const [playing, setPlaying] = React.useState(false)
    const [error, setError] = React.useState<string | undefined>(undefined)

    const [frameCount, setFrameCount] = React.useState(0)
    const [currentFrameIndex, setCurrentFrameIndex] = React.useState(0)
    const [currentPercent, setCurrentPercent] = React.useState(0)
    const [currentFrameCanvas, setCurrentFrameCanvas] = React.useState<HTMLCanvasElement | OffscreenCanvas | null>(null)

    const [width, setWidth] = React.useState(0)
    const [height, setHeight] = React.useState(0)
    const sizeRef = React.useRef<{ width: number; height: number }>({ width: 0, height: 0 })

    const playbackRateRef = React.useRef(1.0)
    React.useEffect(() => {
        playbackRateRef.current = Math.max(1e-6, playbackRate || 1.0)
    }, [playbackRate])

    const opIdRef = React.useRef(0)
    const readyRef = React.useRef(false)
    const playingRef = React.useRef(false)
    const resumeAfterVisibilityRef = React.useRef(false)

    const frameCountRef = React.useRef(0)
    const currentFrameRef = React.useRef(0)

    React.useEffect(() => {
        currentFrameRef.current = currentFrameIndex
    }, [currentFrameIndex])

    const inputRef = React.useRef<Input | null>(null)
    const videoTrackRef = React.useRef<any | null>(null)
    const sinkRef = React.useRef<CanvasSink | null>(null)
    const iterRef = React.useRef<AsyncGenerator<WrappedCanvas | null, void, unknown> | null>(null)

    const ptsRef = React.useRef<number[]>([])
    const durRef = React.useRef<number[]>([])
    const isKeyRef = React.useRef<boolean[]>([])

    const behindFramesRef = React.useRef<number>(0)
    const aheadFramesRef = React.useRef<number>(0)

    const cacheRef = React.useRef<Map<number, WrappedCanvas>>(new Map())
    const cacheStartRef = React.useRef<number>(0)
    const cacheEndRef = React.useRef<number>(-1)

    const resetDecodeState = React.useCallback(async () => {
        const it = iterRef.current
        iterRef.current = null
        try {
            await it?.return?.()
        } catch {}

        cacheRef.current.clear()
        cacheStartRef.current = 0
        cacheEndRef.current = -1
    }, [])

    const dispose = React.useCallback(async () => {
        opIdRef.current++
        readyRef.current = false
        playingRef.current = false

        setPlaying(false)
        setReady(false)
        setLoading(false)
        setSeeking(false)
        setError(undefined)

        frameCountRef.current = 0
        setFrameCount(0)
        setCurrentFrameIndex(0)
        setCurrentPercent(0)
        setCurrentFrameCanvas(null)
        setWidth(0)
        setHeight(0)
        sizeRef.current = { width: 0, height: 0 }

        await resetDecodeState()

        sinkRef.current = null
        videoTrackRef.current = null

        const input = inputRef.current as any
        inputRef.current = null
        try {
            input?.dispose?.()
        } catch {}

        ptsRef.current = []
        durRef.current = []
        isKeyRef.current = []
    }, [resetDecodeState])

    const evictOutsideWindow = React.useCallback((centerIndex: number) => {
        const n = frameCountRef.current
        if (n <= 0) return

        const behind = behindFramesRef.current
        const ahead = aheadFramesRef.current
        const min = clamp(centerIndex - behind, 0, n - 1)
        const max = clamp(centerIndex + ahead, 0, n - 1)

        while (cacheStartRef.current < min) {
            cacheRef.current.delete(cacheStartRef.current)
            cacheStartRef.current++
        }
        while (cacheEndRef.current > max) {
            cacheRef.current.delete(cacheEndRef.current)
            cacheEndRef.current--
        }
    }, [])

    const drawFrameInternal = React.useCallback((idx: number, wc: WrappedCanvas) => {
        const n = frameCountRef.current
        const p = n > 1 ? clamp(idx / (n - 1), 0, 1) : 0
        currentFrameRef.current = idx
        setCurrentFrameCanvas(wc.canvas)

        setCurrentFrameIndex(idx)
        setCurrentPercent(p)
    }, [])

    const prevKeyframeIndex = React.useCallback((idx: number) => {
        const isKey = isKeyRef.current
        for (let i = idx; i >= 0; i--) if (isKey[i]) return i
        return 0
    }, [])

    const getSeekStartIndex = React.useCallback(
        (idx: number) => {
            const n = frameCountRef.current
            if (n <= 0) return 0
            const behind = behindFramesRef.current
            const startCandidate = clamp(idx - behind, 0, n - 1)
            return prevKeyframeIndex(startCandidate)
        },
        [prevKeyframeIndex]
    )

    const shouldReuseForwardIterator = React.useCallback(
        (idx: number) => {
            if (!iterRef.current) return false
            if (cacheRef.current.has(idx)) return true
            if (idx < cacheStartRef.current) return false
            if (idx <= cacheEndRef.current) return true

            const sequentialDistance = idx - cacheEndRef.current
            const restartDistance = idx - getSeekStartIndex(idx)
            const nearbyForwardSeek = sequentialDistance <= Math.max(2, aheadFramesRef.current + 1)

            return nearbyForwardSeek || sequentialDistance <= restartDistance
        },
        [getSeekStartIndex]
    )

    const startIteratorAt = React.useCallback(async (startPts: number, opId: number) => {
        try {
            await iterRef.current?.return?.()
        } catch {}
        iterRef.current = null

        const sink = sinkRef.current
        if (!sink) throw new Error('Video sink not initialized.')
        if (opId !== opIdRef.current) return

        // Use range iterator (never yields "null" placeholders for missing exact timestamps).
        iterRef.current = sink.canvases(startPts)
    }, [])

    const pumpUntil = React.useCallback(
        async (targetEnd: number, opId: number, centerIndex: number) => {
            const pts = ptsRef.current
            const n = pts.length

            while (cacheEndRef.current < targetEnd) {
                if (opId !== opIdRef.current) return

                const it = iterRef.current
                if (!it) return

                const next = await it.next()
                if (opId !== opIdRef.current) return
                if (next.done || !next.value) break

                const wc = next.value

                const idx = cacheEndRef.current + 1
                if (idx < 0 || idx >= n) break

                cacheRef.current.set(idx, wc)
                cacheEndRef.current = idx
                if (cacheStartRef.current > idx) cacheStartRef.current = idx

                evictOutsideWindow(centerIndex)
            }
        },
        [evictOutsideWindow]
    )

    const ensureFrameInCache = React.useCallback(
        async (idx: number, opId: number) => {
            if (cacheRef.current.has(idx)) return

            const n = frameCountRef.current
            if (n <= 0) return

            if (shouldReuseForwardIterator(idx)) {
                await pumpUntil(clamp(idx + aheadFramesRef.current, 0, n - 1), opId, idx)
                if (cacheRef.current.has(idx)) return
            }

            const ahead = aheadFramesRef.current
            const startIdx = getSeekStartIndex(idx)

            cacheRef.current.clear()
            cacheStartRef.current = startIdx
            cacheEndRef.current = startIdx - 1

            await startIteratorAt(ptsRef.current[startIdx]!, opId)
            await pumpUntil(clamp(idx + ahead, 0, n - 1), opId, idx)
        },
        [getSeekStartIndex, pumpUntil, shouldReuseForwardIterator, startIteratorAt]
    )

    const seekFrameInternal = React.useCallback(
        async (i: number, opId: number) => {
            const n = frameCountRef.current
            if (n <= 0) return

            const idx = clamp(i, 0, n - 1)
            await ensureFrameInCache(idx, opId)
            if (opId !== opIdRef.current) return

            const wc = cacheRef.current.get(idx)
            if (!wc) throw new Error(`Failed to decode frame ${idx}.`)

            evictOutsideWindow(idx)
            drawFrameInternal(idx, wc)
        },
        [ensureFrameInCache, evictOutsideWindow, drawFrameInternal]
    )

    const playLoop = React.useCallback(async () => {
        const opId = opIdRef.current
        while (playingRef.current && opId === opIdRef.current) {
            const nextIdx = currentFrameRef.current + 1
            if (nextIdx >= frameCountRef.current) {
                playingRef.current = false
                setPlaying(false)
                break
            }

            try {
                await seekFrameInternal(nextIdx, opId)
            } catch (e: any) {
                if (opId === opIdRef.current) setError(String(e?.message ?? e))
                playingRef.current = false
                setPlaying(false)
                break
            }

            const d = durRef.current[nextIdx] ?? 1 / 30
            const rate = playbackRateRef.current || 1.0
            await sleep(Math.max(0, (d * 1000) / rate))
        }
    }, [seekFrameInternal])

    const play = React.useCallback(() => {
        if (!readyRef.current) return
        if (playingRef.current) return
        playingRef.current = true
        setPlaying(true)
        void playLoop()
    }, [playLoop])

    const pause = React.useCallback(() => {
        playingRef.current = false
        setPlaying(false)
    }, [])

    const togglePlay = React.useCallback(() => {
        if (playingRef.current) pause()
        else play()
    }, [pause, play])

    const seekFrame = React.useCallback(
        async (i: number, playAfter?: boolean) => {
            if (!readyRef.current) return

            const effectivePlayAfter = playAfter ?? playingRef.current

            const opId = ++opIdRef.current

            playingRef.current = false
            setPlaying(false)

            setSeeking(true)
            try {
                await seekFrameInternal(i, opId)
            } catch (e: any) {
                if (opId === opIdRef.current) setError(String(e?.message ?? e))
            } finally {
                if (opId === opIdRef.current) setSeeking(false)
            }

            if (effectivePlayAfter && opId === opIdRef.current) {
                play()
            }
        },
        [seekFrameInternal, play]
    )

    const seekPercent = React.useCallback(
        async (p: number, playAfter?: boolean) => {
            if (!readyRef.current) return
            const n = frameCountRef.current
            if (n <= 0) return
            const idx = clamp(Math.round(clamp(p, 0, 1) * (n - 1)), 0, n - 1)
            await seekFrame(idx, playAfter)
        },
        [seekFrame]
    )

    const stepFrames = React.useCallback(
        async (delta: number) => {
            if (!readyRef.current) return
            playingRef.current = false
            setPlaying(false)
            await seekFrame(currentFrameRef.current + delta, false)
        },
        [seekFrame]
    )

    const load = React.useCallback(
        async (file: File) => {
            await dispose()
            const opId = ++opIdRef.current

            setLoading(true)
            setError(undefined)

            try {
                const fileLabel = `${file.name || 'unnamed'} (${file.type || 'unknown type'}, ${file.size} bytes)`
                const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
                inputRef.current = input

                const videoTrack = await input.getPrimaryVideoTrack()
                if (!videoTrack) throw new Error(`No video track found for ${fileLabel}.`)
                if (videoTrack.codec === null) throw new Error(`Unsupported video codec for ${fileLabel}.`)
                const codec = String(videoTrack.codec)
                if (!(await videoTrack.canDecode()))
                    throw new Error(`Unable to decode video track (${codec}) for ${fileLabel}.`)

                videoTrackRef.current = videoTrack

                const displayW = videoTrack.displayWidth
                const displayH = videoTrack.displayHeight
                sizeRef.current = { width: displayW, height: displayH }
                setWidth(displayW)
                setHeight(displayH)

                const packets = await getSortedVideoPacketMeta(videoTrack)

                if (opId !== opIdRef.current) return
                if (packets.length === 0) throw new Error(`No presentable packets found for ${fileLabel} (${codec}).`)

                const pts = packets.map(p => p.ts)
                const dur = packets.map(p => p.dur)
                const isKey = packets.map(p => p.key)

                ptsRef.current = pts
                durRef.current = dur
                isKeyRef.current = isKey

                frameCountRef.current = pts.length
                setFrameCount(pts.length)

                const stats = await videoTrack.computePacketStats(100)
                const fpsApprox = stats.averagePacketRate || 30
                behindFramesRef.current = Math.max(0, Math.ceil(behindSeconds * fpsApprox))
                aheadFramesRef.current = Math.max(0, Math.ceil(aheadSeconds * fpsApprox))

                const windowSize = behindFramesRef.current + aheadFramesRef.current + 5
                const poolSize = windowSize + 10

                sinkRef.current = new CanvasSink(videoTrack, {
                    poolSize,
                    fit: 'contain',
                    alpha: await videoTrack.canBeTransparent(),
                })

                readyRef.current = true
                setReady(true)

                await seekFrameInternal(0, opId)
            } catch (e: any) {
                if (opId === opIdRef.current) setError(String(e?.message ?? e))
            } finally {
                if (opId === opIdRef.current) setLoading(false)
            }
        },
        [dispose, behindSeconds, aheadSeconds, seekFrameInternal]
    )

    React.useEffect(() => {
        return () => {
            void dispose()
        }
    }, [dispose])

    React.useEffect(() => {
        const onVisibilityChange = () => {
            if (!readyRef.current) return
            if (document.visibilityState === 'hidden') {
                resumeAfterVisibilityRef.current = playingRef.current
                pause()
                return
            }

            const playAfter = resumeAfterVisibilityRef.current
            resumeAfterVisibilityRef.current = false
            void resetDecodeState().then(() => seekFrame(currentFrameRef.current, playAfter))
        }

        document.addEventListener('visibilitychange', onVisibilityChange)
        return () => document.removeEventListener('visibilitychange', onVisibilityChange)
    }, [pause, resetDecodeState, seekFrame])

    const ended = React.useMemo(
        () => frameCount > 0 && currentFrameIndex >= frameCount - 1,
        [frameCount, currentFrameIndex]
    )

    return {
        ready,
        loading,
        seeking,
        playing,
        ended,
        error,
        frameCount,
        currentFrameIndex,
        currentPercent,
        currentFrameCanvas,
        width,
        height,
        load,
        dispose,
        play,
        pause,
        togglePlay,
        seekPercent,
        seekFrame,
        stepFrames,
    }
}
