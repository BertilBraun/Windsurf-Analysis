import React from 'react'
import { PlayerState, findNearestDetectionByTime } from './state'

type Props = {
    state: PlayerState
    videoRef: React.RefObject<HTMLVideoElement>
    onEnterDetailed: (trackId: number) => void
}

export const VideoOverlay: React.FC<Props> = ({ state, videoRef, onEnterDetailed }) => {
    const maxDeltaSec = 0.2
    const [hoveredTrack, setHoveredTrack] = React.useState<number | null>(null)
    const [zoom, setZoom] = React.useState(1)
    const [offset, setOffset] = React.useState<{ x: number; y: number }>({ x: 0, y: 0 })

    if (state.mode !== 'overview') return null

    const bboxes = React.useMemo(() => {
        const v = videoRef.current
        if (!v) return [] as Array<{ x: number; y: number; w: number; h: number; id: number }>
        const baseW = v.clientWidth
        const baseH = v.clientHeight
        const width = baseW * zoom
        const height = baseH * zoom
        const boxes: Array<{ x: number; y: number; w: number; h: number; id: number }> = []

        for (const t of state.tracks) {
            if (!state.visibleTrackIds.has(t.track_id)) continue
            const arr = state.detectionTimesByTrack.get(t.track_id) || []
            const nearest = findNearestDetectionByTime(arr, state.currentTimeSec, maxDeltaSec)
            if (!nearest) continue
            const [x1p, y1p, x2p, y2p] = nearest.detection.bbox
            const x1 = Math.round(x1p * width + offset.x)
            const y1 = Math.round(y1p * height + offset.y)
            const x2 = Math.round(x2p * width + offset.x)
            const y2 = Math.round(y2p * height + offset.y)
            boxes.push({ x: x1, y: y1, w: Math.max(1, x2 - x1), h: Math.max(1, y2 - y1), id: t.track_id })
        }

        return boxes
    }, [
        state.currentTimeSec,
        state.tracks,
        state.visibleTrackIds,
        state.detectionTimesByTrack,
        state.video.height,
        state.video.width,
        videoRef,
        state.mode,
        state.currentTrackId,
        zoom,
        offset.x,
        offset.y,
    ])

    React.useEffect(() => {
        const v = videoRef.current
        if (!v) return
        const onWheel = (e: WheelEvent) => {
            if (state.mode !== 'overview') return
            e.preventDefault()
            const factor = 1 + (e.deltaY < 0 ? 0.1 : -0.1)
            const nz = Math.max(0.25, Math.min(8, zoom * factor))
            if (Math.abs(nz - zoom) < 1e-3) return
            setZoom(nz)
        }
        v.addEventListener('wheel', onWheel, { passive: false })
        return () => v.removeEventListener('wheel', onWheel as any)
    }, [videoRef.current, zoom, state.mode])

    return (
        <div className="absolute left-0 top-0 right-0 bottom-0 z-2">
            {bboxes.map(b => (
                <div
                    key={b.id}
                    className="absolute"
                    style={{
                        left: b.x,
                        top: b.y,
                        width: b.w,
                        height: b.h,
                        border:
                            '2px solid ' +
                            (hoveredTrack === b.id ? '#10b981' : state.currentTrackId === b.id ? '#f59e0b' : '#ef4444'),
                        boxSizing: 'border-box',
                        cursor: 'pointer',
                    }}
                    onMouseEnter={() => setHoveredTrack(b.id)}
                    onMouseLeave={() => setHoveredTrack(null)}
                    onClick={() => onEnterDetailed(b.id)}
                >
                    {b.id}
                </div>
            ))}
        </div>
    )
}
