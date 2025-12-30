import React from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from '../components/Button'

export type DrawTool = 'freehand' | 'line'

export const DRAW_WIDTH_OPTIONS = [2, 3, 5, 8, 14, 20]
export const DRAW_COLOR_OPTIONS = ['#f97316', '#22c55e', '#3b82f6', '#ef4444', '#facc15']

const WIDTH_PREVIEW_MIN = 8
const WIDTH_PREVIEW_MAX = 24

type DrawOverlayProps = {
    drawTool: DrawTool
    onDrawToolChange: (tool: DrawTool) => void
    drawWidth: number
    onDrawWidthChange: (width: number) => void
    drawColor: string
    onDrawColorChange: (color: string) => void
    onClearAnnotations: () => void
    hasVisibleAnnotations: boolean
}

export const DrawOverlay: React.FC<DrawOverlayProps> = ({
    drawTool,
    onDrawToolChange,
    drawWidth,
    onDrawWidthChange,
    drawColor,
    onDrawColorChange,
    onClearAnnotations,
    hasVisibleAnnotations,
}) => {
    const { t } = useTranslation()

    return (
        <div className="absolute left-3 top-3 z-10 rounded-md bg-black/60 border border-gray-700 text-gray-100 text-xs px-3 py-2 space-y-3">
            <div>
                <div className="text-[11px] uppercase tracking-wide text-gray-300">{t('player.canvas.draw.tool')}</div>
                <div className="mt-2 flex items-center gap-1">
                    <Button
                        type="button"
                        variant={drawTool === 'freehand' ? 'outline' : 'inverse'}
                        size="sm"
                        onClick={() => onDrawToolChange('freehand')}
                        text={t('player.canvas.draw.freehand')}
                    />
                    <Button
                        type="button"
                        variant={drawTool === 'line' ? 'outline' : 'inverse'}
                        size="sm"
                        onClick={() => onDrawToolChange('line')}
                        text={t('player.canvas.draw.line')}
                    />
                </div>
            </div>
            <div>
                <div className="text-[11px] uppercase tracking-wide text-gray-300">{t('player.canvas.draw.width')}</div>
                <div className="mt-2 flex items-center gap-2">
                    {DRAW_WIDTH_OPTIONS.map((option, index) => {
                        const previewSize =
                            (index / (DRAW_WIDTH_OPTIONS.length - 1)) * (WIDTH_PREVIEW_MAX - WIDTH_PREVIEW_MIN) +
                            WIDTH_PREVIEW_MIN
                        const buttonSize = previewSize
                        return (
                            <Button
                                key={option}
                                type="button"
                                variant="unstyled"
                                size="none"
                                onClick={() => onDrawWidthChange(option)}
                                className={`flex items-center justify-center rounded-full transition ${
                                    drawWidth === option
                                        ? 'ring-2 ring-brand-600'
                                        : 'ring-1 ring-white/30 hover:ring-white/70'
                                }`}
                                style={{ width: buttonSize, height: buttonSize }}
                                aria-label={t('player.canvas.draw.widthLabel', { value: option })}
                            >
                                <span
                                    className="block rounded-full bg-white"
                                    style={{ width: previewSize, height: previewSize }}
                                />
                            </Button>
                        )
                    })}
                </div>
            </div>
            <div>
                <div className="text-[11px] uppercase tracking-wide text-gray-300">{t('player.canvas.draw.color')}</div>
                <div className="mt-2 flex items-center gap-2">
                    {DRAW_COLOR_OPTIONS.map(option => (
                        <Button
                            key={option}
                            type="button"
                            variant="unstyled"
                            size="none"
                            onClick={() => onDrawColorChange(option)}
                            className={`h-6 w-6 rounded-full transition ${
                                drawColor === option ? 'ring-2 ring-white' : 'ring-1 ring-white/30 hover:ring-white/70'
                            }`}
                            style={{ backgroundColor: option }}
                            aria-label={t('player.canvas.draw.colorLabel', { value: option })}
                        />
                    ))}
                </div>
            </div>
            <Button
                size="sm"
                text={t('player.canvas.draw.clearFrame')}
                onClick={onClearAnnotations}
                disabled={!hasVisibleAnnotations}
            />
        </div>
    )
}
