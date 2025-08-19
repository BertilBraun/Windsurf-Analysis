import React from 'react'
import MediaInfoFactory from 'mediainfo.js'

export function useVideoFps() {
    const [fps, setFps] = React.useState<number | undefined>(undefined)

    const analyzeFile = React.useCallback(async (file: File) => {
        try {
            const mediaInfo = await MediaInfoFactory({
                format: 'object',
                locateFile: path => `https://unpkg.com/mediainfo.js/dist/${path}`,
            })
            const arrayBuffer = await file.arrayBuffer()
            const result = await mediaInfo.analyzeData(
                () => arrayBuffer.byteLength,
                (chunkSize: number, offset: number) => new Uint8Array(arrayBuffer, offset, chunkSize)
            )
            const tracks = result.media?.track as Array<Record<string, unknown>> | undefined
            const videoTrack = tracks?.find(t => t['@type'] === 'Video') as Record<string, unknown> | undefined
            if (!videoTrack) {
                setFps(undefined)
                return
            }
            const direct = parseFloat(videoTrack.FrameRate as string)
            const num = videoTrack.FrameRate_Num
            const den = videoTrack.FrameRate_Den
            const frac = num && den ? Number(num) / Number(den) : undefined
            const computed =
                Number.isFinite(direct) && direct > 0
                    ? direct
                    : Number.isFinite(frac as number) && (frac as number) > 0
                    ? (frac as number)
                    : undefined
            setFps(computed)
        } catch {
            setFps(undefined)
        }
    }, [])

    const reset = React.useCallback(() => setFps(undefined), [])

    const frameDuration = React.useMemo(() => (fps && Number.isFinite(fps) ? 1 / fps : 1 / 120), [fps])

    return { fps, frameDuration, analyzeFile, reset }
}
