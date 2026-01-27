/**
 * @module ControlsBar
 * Provides a user interface for controlling playback, speed, zoom, and track export.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from '../components/Button'

/**
 * Props for the ControlsBar component.
 */
export interface ControlsBarProps {
    /** Callback triggered when the play/pause button is clicked. */
    onPlayPause: () => void
    /** Callback triggered to decrease playback speed. */
    onSpeedDown: () => void
    /** Callback triggered to increase playback speed. */
    onSpeedUp: () => void
    /** Indicates if the player is currently playing. */
    isPlaying: boolean
    /** Current playback speed multiplier. */
    speed: number
    /** Current zoom level of the player view. */
    zoom: number
    /** Optional callback to trigger track export. */
    onExportTrack?: () => void
    /** Whether the export button should be visible. */
    exportVisible?: boolean
    /** Whether the export button is interactive. */
    exportEnabled?: boolean
    /** Indicates if an export process is currently active. */
    isExporting?: boolean
    /** Current export progress percentage (0-100). */
    exportProgressPct?: number | null
}

/**
 * A toolbar component containing playback controls, status indicators, and export functionality.
 *
 * @param props - The component props.
 */
export const ControlsBar: React.FC<ControlsBarProps> = ({
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
