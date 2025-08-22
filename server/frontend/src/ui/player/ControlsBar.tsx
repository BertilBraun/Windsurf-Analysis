import React from 'react'
import { Button } from '../components/Button'

export const ControlsBar: React.FC<{
    onPlayPause: () => void
    onSpeedDown: () => void
    onSpeedUp: () => void
    isPlaying: boolean
    speed: number
    zoom: number
}> = ({ onPlayPause, onSpeedDown, onSpeedUp, isPlaying, speed, zoom }) => {
    return (
        <div className="flex items-center gap-2">
            <Button onClick={onPlayPause} text={isPlaying ? 'Pause' : 'Play'} />
            <div className="flex-1" />
            <div className="text-sm">Zoom: {zoom.toFixed(2)}x</div>
            <Button onClick={onSpeedDown} text="- Speed" />
            <div className="text-sm">Speed: {speed.toFixed(2)}x</div>
            <Button onClick={onSpeedUp} text="+ Speed" />
        </div>
    )
}
