/**
 * @module ControlsBar
 * Provides a user interface for controlling playback, speed, and track export.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from '../components/Button'

function cx(...parts: Array<string | undefined | null | false>) {
    return parts.filter(Boolean).join(' ')
}

function formatMinSec(seconds: number | null | undefined) {
    if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '--:--'
    const s = Math.floor(seconds)
    const mm = Math.floor(s / 60)
    const ss = s % 60
    return `${mm}:${String(ss).padStart(2, '0')}`
}

/**
 * Props for the ControlsBar component.
 */
export interface ControlsBarProps {
    /** Callback triggered when the play/pause button is clicked. */
    onPlayPause: () => void
    /** Indicates if the player is currently playing. */
    isPlaying: boolean
    /** Current playback time in seconds (approx). */
    currentTimeSeconds?: number | null
    /** Total video duration in seconds (approx). */
    totalTimeSeconds?: number | null
    /** Current playback speed multiplier. */
    speed: number
    /** Allowed playback speeds for selection. */
    speedRates: number[]
    /** Callback triggered to set playback speed. */
    onSetSpeed: (speed: number) => void
    /** Optional callback to trigger track export. */
    onExportTrack?: () => void
    /** Whether the export button should be visible. */
    exportVisible?: boolean
}

/**
 * A toolbar component containing playback controls, status indicators, and export functionality.
 *
 * @param props - The component props.
 */
export const ControlsBar: React.FC<ControlsBarProps> = ({
    onPlayPause,
    isPlaying,
    currentTimeSeconds,
    totalTimeSeconds,
    speed,
    speedRates,
    onSetSpeed,
    onExportTrack,
    exportVisible,
}) => {
    const { t } = useTranslation()
    const [speedMenuOpen, setSpeedMenuOpen] = React.useState(false)
    const speedMenuRef = React.useRef<HTMLDivElement | null>(null)

    React.useEffect(() => {
        const onPointerDown = (event: MouseEvent) => {
            const el = speedMenuRef.current
            if (!el) return
            if (el.contains(event.target as Node)) return
            setSpeedMenuOpen(false)
        }
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') setSpeedMenuOpen(false)
        }
        document.addEventListener('mousedown', onPointerDown)
        document.addEventListener('keydown', onKeyDown)
        return () => {
            document.removeEventListener('mousedown', onPointerDown)
            document.removeEventListener('keydown', onKeyDown)
        }
    }, [])

    return (
        <div className="flex items-center gap-2">
            <Button
                onClick={onPlayPause}
                text={isPlaying ? t('player.controlsBar.pause') : t('player.controlsBar.play')}
            />
            <div className="text-sm text-gray-200 tabular-nums">
                {formatMinSec(currentTimeSeconds)} / {formatMinSec(totalTimeSeconds)}
            </div>
            <div className="flex-1" />
            <div ref={speedMenuRef} className="relative">
                <Button
                    type="button"
                    aria-haspopup="menu"
                    aria-expanded={speedMenuOpen}
                    onClick={() => setSpeedMenuOpen(open => !open)}
                    text={t('player.controlsBar.speed', { value: speed.toFixed(2) })}
                />
                {speedMenuOpen && (
                    <div
                        role="menu"
                        aria-label={t('player.controlsBar.speed', { value: speed.toFixed(2) })}
                        className="absolute right-0 bottom-full mb-2 w-28 rounded-md border border-slate-200 bg-white shadow-lg z-50 overflow-hidden"
                    >
                        {speedRates.map(rate => {
                            const isActive = Math.abs(rate - speed) < 1e-9
                            const label = `${Number.isInteger(rate) ? rate.toFixed(0) : String(rate)}x`
                            return (
                                <Button
                                    key={rate}
                                    type="button"
                                    variant="unstyled"
                                    size="none"
                                    role="menuitem"
                                    onClick={() => {
                                        onSetSpeed(rate)
                                        setSpeedMenuOpen(false)
                                    }}
                                    className={cx(
                                        'w-full px-3 py-2 text-xs text-left text-slate-700 hover:bg-slate-50',
                                        isActive ? 'bg-slate-100 text-slate-900' : undefined
                                    )}
                                >
                                    {label}
                                </Button>
                            )
                        })}
                    </div>
                )}
            </div>
            {exportVisible && onExportTrack && (
                <>
                    <div className="w-px h-5 bg-gray-600 mx-1" />
                    <Button
                        onClick={onExportTrack}
                        title={t('player.controlsBar.exportTitle')}
                        text={t('player.controlsBar.export')}
                    />
                </>
            )}
        </div>
    )
}
