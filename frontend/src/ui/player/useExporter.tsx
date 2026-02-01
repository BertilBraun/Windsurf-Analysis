import React from 'react'
import { useTranslation } from 'react-i18next'
import { Track } from '../types'
import { clamp } from '../utils/clamp'
import { trackEvent } from '../utils/analytics'
import { buildExportFilename, exportTrackMp4 } from './export'
import { ExportOverlay, ExportResult } from './ExportOverlay'

type UseExporterParams = {
    sourceFile: File | null
    frameCount: number | null
}

type UseExporterResult = {
    isExporting: boolean
    exportError: string | null
    exportTrack: (args: { track: Track; dominantOrientationDeg: number; localRelativePath?: string | null }) => Promise<void>
    reset: () => void
    overlay: React.ReactNode
}

export function useExporter(params: UseExporterParams): UseExporterResult {
    const { t } = useTranslation()
    const { sourceFile, frameCount } = params

    const [exportError, setExportError] = React.useState<string | null>(null)
    const [exportResult, setExportResult] = React.useState<ExportResult | null>(null)
    const [isExporting, setIsExporting] = React.useState(false)
    const [exportProgressPct, setExportProgressPct] = React.useState<number | null>(null)
    const sessionRef = React.useRef(0)

    const reset = React.useCallback(() => {
        sessionRef.current += 1
        setIsExporting(false)
        setExportProgressPct(null)
        setExportResult(null)
        setExportError(null)
    }, [])

    const exportTrack = React.useCallback(
        async (args: { track: Track; dominantOrientationDeg: number; localRelativePath?: string | null }) => {
            if (isExporting) return
            const { track, dominantOrientationDeg, localRelativePath } = args
            const session = sessionRef.current + 1
            sessionRef.current = session
            setExportError(null)
            setExportResult(null)
            setIsExporting(true)
            setExportProgressPct(0)

            try {
                const file = sourceFile
                const fc = frameCount
                if (!file || typeof fc !== 'number') throw new Error(t('player.canvas.export.errors.notReady'))

                trackEvent('export_track_start', { track_id: track.track_id })

                const padSec = 0.25
                const startSec = Math.max(0, track.start_time_seconds - padSec)
                const endSec = track.start_time_seconds + track.duration_seconds + padSec
                if (!(endSec > startSec + 1e-3)) throw new Error(t('player.canvas.export.errors.trackTooShort'))

                const outBlob = await exportTrackMp4({
                    file,
                    frameCount: fc,
                    dominantOrientationDeg,
                    trackId: track.track_id,
                    trackDetections: track.detections,
                    startSec,
                    endSec,
                    onProgress: prog01 => setExportProgressPct(clamp(prog01 * 100, 0, 100)),
                })

                const filename = buildExportFilename({
                    sourceFileName: file.name,
                    localRelativePath,
                    trackId: track.track_id,
                    startSec,
                    endSec,
                })

                if (sessionRef.current !== session) return
                setExportResult({ blob: outBlob, filename })
                trackEvent('export_track_success', { track_id: track.track_id })
            } catch (e: any) {
                const fallback = t('player.canvas.export.errors.failed')
                const message = String(e?.message || e || fallback)
                if (sessionRef.current !== session) return
                setExportError(message)
                trackEvent('export_track_failed', { message })
            } finally {
                if (sessionRef.current !== session) return
                setIsExporting(false)
                setExportProgressPct(null)
            }
        },
        [frameCount, isExporting, sourceFile, t]
    )

    const overlay = (
        <ExportOverlay
            isExporting={isExporting}
            exportProgressPct={exportProgressPct}
            exportResult={exportResult}
            onClearExportResult={() => {
                setExportResult(null)
                setExportError(null)
            }}
        />
    )

    return {
        isExporting,
        exportError,
        exportTrack,
        reset,
        overlay,
    }
}
