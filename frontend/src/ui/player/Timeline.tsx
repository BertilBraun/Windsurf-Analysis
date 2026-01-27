/**
 * @file Timeline.tsx
 * @description A timeline slider component for media playback control and seeking.
 */

import React from 'react'
import { assert } from '../utils/assert'
import { clamp } from '../utils/clamp'

/**
 * A visual timeline slider that allows users to view and seek playback progress.
 *
 * This component handles pointer interactions to calculate seek positions and
 * renders a progress bar based on the current playback state.
 */
export const Timeline: React.FC<{
    /** Callback invoked when the user seeks to a specific percentage (0 to 1). */
    onSeekPercent: (currentProgressPercent: number) => void
    /** The current playback progress as a decimal between 0 and 1. */
    currentProgressPercent: number
}> = ({ onSeekPercent, currentProgressPercent }) => {
    assert(0 <= currentProgressPercent && currentProgressPercent <= 1, 'currentProgressPercent must be between 0 and 1')
    const percent = currentProgressPercent * 100

    const isDraggingRef = React.useRef(false)
    const activePointerIdRef = React.useRef<number | null>(null)
    const rootRef = React.useRef<HTMLDivElement | null>(null)

    const seekFromClientX = React.useCallback(
        (clientX: number) => {
            const el = rootRef.current
            if (!el) return
            const rect = el.getBoundingClientRect()
            const px = Math.max(0, Math.min(rect.width, clientX - rect.left))
            const p = rect.width > 0 ? px / rect.width : 0
            const currentProgressPercent = clamp(p, 0, 1)
            onSeekPercent(currentProgressPercent)
        },
        [onSeekPercent]
    )

    const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
        if (e.button !== 0) return
        e.preventDefault()
        isDraggingRef.current = true
        activePointerIdRef.current = e.pointerId
        rootRef.current?.setPointerCapture?.(e.pointerId)
        seekFromClientX(e.clientX)
    }

    const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
        if (!isDraggingRef.current) return
        if (activePointerIdRef.current !== e.pointerId) return
        e.preventDefault()
        seekFromClientX(e.clientX)
    }

    const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
        if (activePointerIdRef.current !== e.pointerId) return
        isDraggingRef.current = false
        activePointerIdRef.current = null
        try {
            rootRef.current?.releasePointerCapture?.(e.pointerId)
        } catch {}
    }

    return (
        <div
            ref={rootRef}
            className="relative h-2 bg-gray-200 cursor-pointer select-none touch-none"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onPointerLeave={endDrag}
        >
            <div className="absolute left-0 top-0 bottom-0 bg-gray-400" style={{ width: `${percent}%` }} />
            <div
                className="absolute top-0 bottom-0 bg-yellow-500"
                style={{ left: `calc(${percent}% - 1px)`, width: 2 }}
            />
        </div>
    )
}
