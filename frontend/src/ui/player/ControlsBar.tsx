import React from 'react'
import { useTranslation } from 'react-i18next'
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
    const { t } = useTranslation()
    return (
        <div className="flex items-center gap-2">
            <Button
                onClick={onPlayPause}
                text={isPlaying ? t('player.controlsBar.pause') : t('player.controlsBar.play')}
            />
            <div className="flex-1" />
            <div className="text-sm">{t('player.controlsBar.zoom', { value: zoom.toFixed(2) })}</div>
            <Button onClick={onSpeedDown} text={t('player.controlsBar.speedDown')} />
            <div className="text-sm">{t('player.controlsBar.speed', { value: speed.toFixed(2) })}</div>
            <Button onClick={onSpeedUp} text={t('player.controlsBar.speedUp')} />
            {exportVisible && onExportTrack && (
                <>
                    <div className="w-px h-5 bg-gray-600 mx-1" />
                    {typeof exportProgressPct === 'number' && isExporting && (
                        <div className="text-sm tabular-nums">
                            {Math.max(0, Math.min(100, exportProgressPct)).toFixed(0)}%
                        </div>
                    )}
                    <Button
                        onClick={onExportTrack}
                        title={t('player.controlsBar.exportTitle')}
                        text={t('player.controlsBar.export')}
                        disabled={!exportEnabled || !!isExporting}
                    />
                </>
            )}
        </div>
    )
}
