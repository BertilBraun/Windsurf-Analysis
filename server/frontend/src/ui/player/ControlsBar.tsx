import React from 'react'

export const ControlsBar: React.FC<{
    onPlayPause: () => void
    onSpeedDown: () => void
    onSpeedUp: () => void
    isPlaying: boolean
    speed: number
}> = ({ onPlayPause, onSpeedDown, onSpeedUp, isPlaying, speed }) => {
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button onClick={onPlayPause}>{isPlaying ? 'Pause' : 'Play'}</button>
            <div style={{ flex: 1 }} />
            <button onClick={onSpeedDown}>- Speed</button>
            <div style={{ fontSize: 12 }}>Speed: {speed.toFixed(2)}x</div>
            <button onClick={onSpeedUp}>+ Speed</button>
        </div>
    )
}
