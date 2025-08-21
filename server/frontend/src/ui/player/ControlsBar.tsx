import React from 'react'
import { Button } from '../components/Button'
import { KeyboardShortcutsModal } from '../components/KeyboardShortcutsModal'

export const ControlsBar: React.FC<{
    onPlayPause: () => void
    onSpeedDown: () => void
    onSpeedUp: () => void
    isPlaying: boolean
    speed: number
    zoom: number
}> = ({ onPlayPause, onSpeedDown, onSpeedUp, isPlaying, speed, zoom }) => {
    const [showShortcuts, setShowShortcuts] = React.useState<boolean>(false)

    return (
        <div className="flex items-center gap-2">
            <button onClick={onPlayPause}>{isPlaying ? 'Pause' : 'Play'}</button>
            <div className="flex-1" />
            <div className="text-sm">Zoom: {zoom.toFixed(2)}x</div>
            <button onClick={onSpeedDown}>- Speed</button>
            <div className="text-sm">Speed: {speed.toFixed(2)}x</div>
            <button onClick={onSpeedUp}>+ Speed</button>
            <Button onClick={() => setShowShortcuts(true)} title="Keyboard shortcuts" text="Shortcuts" />

            {showShortcuts && <KeyboardShortcutsModal onClose={() => setShowShortcuts(false)} />}
        </div>
    )
}
