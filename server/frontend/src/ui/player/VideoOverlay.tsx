import React from 'react'
import { PlayerState, findNearestDetectionByTime } from './state'

type Props = {
    state: PlayerState
    videoRef: React.RefObject<HTMLVideoElement>
    onEnterDetailed: (trackId: number) => void
    onWheelZoom?: (cx: number, cy: number, deltaY: number) => void
}

export const VideoOverlay: React.FC<Props> = ({ state, videoRef, onEnterDetailed, onWheelZoom }) => {
    const maxDeltaSec = 0.2
    const [hoveredTrack, setHoveredTrack] = React.useState<number | null>(null)

    if (state.mode !== 'overview') return null

    const bboxes = React.useMemo(() => {
        const v = videoRef.current
        if (!v) return [] as Array<{ x: number; y: number; w: number; h: number; id: number }>
        const width = v.clientWidth
        const height = v.clientHeight
        const boxes: Array<{ x: number; y: number; w: number; h: number; id: number }> = []

        for (const t of state.tracks) {
            if (!state.visibleTrackIds.has(t.track_id)) continue
            const arr = state.detectionTimesByTrack.get(t.track_id) || []
            const nearest = findNearestDetectionByTime(arr, state.currentTimeSec, maxDeltaSec)
            if (!nearest) continue
            const [x1p, y1p, x2p, y2p] = nearest.detection.bbox
            const x1 = Math.round(x1p * width)
            const y1 = Math.round(y1p * height)
            const x2 = Math.round(x2p * width)
            const y2 = Math.round(y2p * height)
            boxes.push({ x: x1, y: y1, w: Math.max(1, x2 - x1), h: Math.max(1, y2 - y1), id: t.track_id })
        }

        return boxes
    }, [state.currentTimeSec, state.tracks, state.visibleTrackIds, state.detectionTimesByTrack, videoRef])

    return (
        <div
            className="absolute left-0 top-0 right-0 bottom-0 z-2"
            onWheel={e => onWheelZoom?.(e.clientX, e.clientY, e.deltaY)}
        >
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
