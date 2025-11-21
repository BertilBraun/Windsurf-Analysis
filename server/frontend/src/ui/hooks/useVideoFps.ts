import React from 'react'
import MediaInfoFactory, { VideoTrack } from 'mediainfo.js'

export async function getVideoTrack(file: File): Promise<VideoTrack | undefined> {
    const mediaInfo = await MediaInfoFactory({
        locateFile: (path: string) => `/${path}`,
    })
    const arrayBuffer = await file.arrayBuffer()
    const result = await mediaInfo.analyzeData(
        () => arrayBuffer.byteLength,
        (chunkSize: number, offset: number) => new Uint8Array(arrayBuffer, offset, chunkSize)
    )
    const tracks = result.media?.track
    const videoTrack = tracks?.find(t => t['@type'] === 'Video')
    return videoTrack
}

export function useVideoFps() {
    const [fps, setFps] = React.useState<number | undefined>(undefined)

    const analyzeFile = React.useCallback(async (file: File) => {
        try {
            const videoTrack = await getVideoTrack(file)
            if (!videoTrack) {
                setFps(undefined)
                return
            }
            if (videoTrack.FrameRate) {
                setFps(videoTrack.FrameRate)
                return
            }
            const num = videoTrack.FrameRate_Num
            const den = videoTrack.FrameRate_Den
            if (num && den) {
                setFps(num / den)
                return
            }
            setFps(undefined)
        } catch (error: any) {
            console.error('Error analyzing video file', file.name, 'error:', error.message, 'setting fps to undefined')
            setFps(undefined)
        }
    }, [])

    const reset = React.useCallback(() => setFps(undefined), [])

    const frameDuration = React.useMemo(() => (fps && Number.isFinite(fps) ? 1 / fps : 1 / 120), [fps])

    return { fps, frameDuration, analyzeFile, reset }
}
