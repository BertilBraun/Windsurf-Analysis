import React from 'react'
import { Button } from '../components/Button'

export const ControlsBar: React.FC<{
    onPlayPause: () => void
    onSpeedDown: () => void
    onSpeedUp: () => void
    isPlaying: boolean
    speed: number
    zoom: number
    onExportTrack?: () => void
    exportVisible?: boolean
    exportEnabled?: boolean
    isExporting?: boolean
    exportProgressPct?: number | null
}> = ({
    onPlayPause,
    onSpeedDown,
    onSpeedUp,
    isPlaying,
    speed,
    zoom,
    onExportTrack,
    exportVisible,
    exportEnabled,
    isExporting,
    exportProgressPct,
}) => {
    return (
        <div className="flex items-center gap-2">
            <Button onClick={onPlayPause} text={isPlaying ? 'Pause' : 'Play'} />
            <div className="flex-1" />
            <div className="text-sm">Zoom: {zoom.toFixed(2)}x</div>
            <Button onClick={onSpeedDown} text="- Speed" />
            <div className="text-sm">Speed: {speed.toFixed(2)}x</div>
            <Button onClick={onSpeedUp} text="+ Speed" />
            {exportVisible && onExportTrack && (
                <>
                    <div className="w-px h-5 bg-gray-600 mx-1" />
                    {typeof exportProgressPct === 'number' && isExporting && (
                        <div className="text-sm tabular-nums">
                            {Math.max(0, Math.min(100, exportProgressPct)).toFixed(0)}%
                        </div>
                    )}
                    <Button onClick={onExportTrack} text="Export" disabled={!exportEnabled || !!isExporting} />
                </>
            )}
        </div>
    )
}
