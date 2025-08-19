import React from 'react'
import { PlayerState } from './state'

type Props = {
    state: PlayerState
    videoRef: React.RefObject<HTMLVideoElement>
    onEnterDetailed: (trackId: number) => void
    onWheelZoom?: (cx: number, cy: number, deltaY: number) => void
}

export const VideoOverlay: React.FC<Props> = ({ state, videoRef, onEnterDetailed, onWheelZoom }) => {
    const [hoveredTrack, setHoveredTrack] = React.useState<number | null>(null)

    if (state.mode !== 'overview') return null

    const bboxes = React.useMemo(() => {
        const v = videoRef.current
        if (!v) return [] as Array<{ x: number; y: number; w: number; h: number; id: number }>
        const elemW = v.clientWidth
        const elemH = v.clientHeight
        const vidW = v.videoWidth
        const vidH = v.videoHeight
        const scale = Math.min(elemW / vidW, elemH / vidH)
        const dispW = vidW * scale
        const dispH = vidH * scale
        const offX = (elemW - dispW) / 2
        const offY = (elemH - dispH) / 2
        const boxes: Array<{ x: number; y: number; w: number; h: number; id: number }> = []

        for (const t of state.tracks) {
            if (!state.visibleTrackIds.has(t.track_id)) continue

            const detection = state.interpolateDetectionByTime(t.track_id, state.currentTimeSec)
            if (!detection) continue
            const [x1p, y1p, x2p, y2p] = detection.bbox

            const x1 = Math.round(offX + x1p * dispW)
            const y1 = Math.round(offY + y1p * dispH)
            const w = Math.max(1, Math.round((x2p - x1p) * dispW))
            const h = Math.max(1, Math.round((y2p - y1p) * dispH))
            boxes.push({ x: x1, y: y1, w, h, id: t.track_id })
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
