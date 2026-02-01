import React from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from '../components/Button'
import { trackEvent } from '../utils/analytics'
import { canShareExport, downloadExport, shareExport } from './export'

export type ExportResult = {
    blob: Blob
    filename: string
    jobId: string
    trackId: number
}

type Props = {
    isExporting: boolean
    exportProgressPct: number | null
    exportResult: ExportResult | null
    onClearExportResult: () => void
}

function isShareTooLargeError(error: unknown): boolean {
    const message = String((error as any)?.message ?? error ?? '').toLowerCase()
    const name = String((error as any)?.name ?? '').toLowerCase()
    return message.includes('too large') || name === 'dataerror'
}

export const ExportOverlay: React.FC<Props> = ({ isExporting, exportProgressPct, exportResult, onClearExportResult }) => {
    const { t } = useTranslation()
    const [shareError, setShareError] = React.useState<string | null>(null)

    React.useEffect(() => {
        setShareError(null)
    }, [exportResult, isExporting])

    const shareSupported = React.useMemo(() => {
        if (!exportResult) return false
        return canShareExport(exportResult.blob, exportResult.filename)
    }, [exportResult])

    const onShare = React.useCallback(async () => {
        const result = exportResult
        if (!result) return

        setShareError(null)
        if (!shareSupported) {
            setShareError(t('player.canvas.export.shareUnsupported'))
            return
        }

        try {
            trackEvent('export_track_share_clicked', { job_id: result.jobId, track_id: result.trackId })
            await shareExport({
                blob: result.blob,
                filename: result.filename,
                text: t('player.canvas.export.shareText'),
                title: 'GybeLock',
            })
            trackEvent('export_track_share_success', { job_id: result.jobId, track_id: result.trackId })
            onClearExportResult()
        } catch (e: any) {
            if (e?.name === 'AbortError') return
            const message = isShareTooLargeError(e)
                ? t('player.canvas.export.shareTooLarge')
                : t('player.canvas.export.shareFailed')
            setShareError(message)
            trackEvent('export_track_share_failed', { job_id: result.jobId, track_id: result.trackId })
        }
    }, [exportResult, onClearExportResult, shareSupported, t])

    const onDownload = React.useCallback(() => {
        const result = exportResult
        if (!result) return
        trackEvent('export_track_download_clicked', { job_id: result.jobId, track_id: result.trackId })
        downloadExport(result.blob, result.filename)
        onClearExportResult()
    }, [exportResult, onClearExportResult])

    const onClose = React.useCallback(() => {
        setShareError(null)
        onClearExportResult()
    }, [onClearExportResult])

    if (!isExporting && !exportResult) return null

    return (
        <div
            className="absolute inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
            onMouseDown={e => e.preventDefault()}
            onClick={e => e.preventDefault()}
            onWheel={e => e.preventDefault()}
        >
            <div className="px-4 py-3 rounded-lg bg-black/60 border border-gray-700 text-gray-100 text-center">
                {isExporting ? (
                    <>
                        <div className="text-base font-semibold">{t('player.canvas.export.overlay.title')}</div>
                        {typeof exportProgressPct === 'number' ? (
                            <div className="mt-1 text-sm tabular-nums">
                                {Math.max(0, Math.min(100, exportProgressPct)).toFixed(0)}%
                            </div>
                        ) : (
                            <div className="mt-1 text-sm">{t('player.canvas.export.overlay.starting')}</div>
                        )}
                    </>
                ) : (
                    <>
                        <div className="text-base font-semibold">{t('player.canvas.export.overlay.readyTitle')}</div>
                        {shareError && <div className="mt-2 text-sm text-red-300">{shareError}</div>}
                        <div className="mt-3 flex items-center justify-center gap-2">
                            <Button variant="primary" onClick={onShare}>
                                {t('player.canvas.export.overlay.share')}
                            </Button>
                            <Button variant="inverse" onClick={onDownload}>
                                {t('player.canvas.export.overlay.download')}
                            </Button>
                            <Button variant="inverse" onClick={onClose}>
                                {t('common.close')}
                            </Button>
                        </div>
                    </>
                )}
            </div>
        </div>
    )
}

