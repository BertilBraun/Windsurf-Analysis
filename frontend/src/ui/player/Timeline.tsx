import React from 'react'
import { PlayerState } from './state'

export const Timeline: React.FC<{
    state: PlayerState
    onSeekTime: (timeSec: number) => void
}> = ({ state, onSeekTime }) => {
    const duration = state.video.durationSeconds
    const percent = (state.currentTimeSec / duration) * 100

    const onClick = (e: React.MouseEvent<HTMLDivElement>) => {
        const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect()
        const px = Math.max(0, Math.min(rect.width, e.clientX - rect.left))
        const p = px / rect.width
        const t = Math.max(0, Math.min(duration, p * duration))
        onSeekTime(t)
    }

    return (
        <div className="relative h-7 bg-gray-200 cursor-pointer" onClick={onClick}>
            <div className="absolute left-0 top-0 bottom-0 bg-gray-400" style={{ width: `${percent}%` }} />
            {state.tracks.map(t => {
                const lx = Math.max(0, t.start_percent) * 100 || 0
                const rx = Math.min(1, t.end_percent) * 100 || 0
                const left = `${lx}%`
                const width = `${Math.max(0.5, rx - lx)}%`
                return (
                    <div
                        key={t.track_id}
                        className="absolute bottom-0"
                        style={{ left, height: 12, width, background: 'rgba(80,160,240,0.55)' }}
                    />
                )
            })}
            <div
                className="absolute top-0 bottom-0 bg-yellow-500"
                style={{ left: `calc(${percent}% - 1px)`, width: 2 }}
            />
        </div>
    )
}
