import React from 'react'
import { AuthorizedFetch, uploadVideoFile } from '../utils/uploader'
import { useSettings } from './useSettings'
import { useLocalFileIndex } from './useLocalFileIndex'
import { type FileFingerprint } from '../utils/localFileIndex'
import { assert } from '../utils/assert'
import type { JobSummary } from '../types'
import { useTabLeader } from './useTabLeader'
import {
    publishIngressScannerState,
    requestIngressRetryFailed,
    subscribeIngressScannerCommands,
    subscribeIngressScannerState,
    type IngressScannerSharedState,
} from '../utils/ingressScannerSync'

export type IngressUploadStatus = 'queued' | 'uploading' | 'done' | 'error' | 'skipped'

export type IngressUploadItem = {
    id: string // sha256 of the original file
    relativePath: string
    progress: number // 0-100
    status: IngressUploadStatus
    error?: string | null
}

const AUTO_RETRY_MS = 15000
const HANDLED_TTL_MS = 15_000
const SKIPPED_TTL_MS = 60 * 60 * 1000

function isTransientUploadError(err: any): boolean {
    const msg = String(err?.message || err || '').toLowerCase()
    if (!msg) return false
    return (
        msg.includes('network') ||
        msg.includes('failed to fetch') ||
        msg.includes('timeout') ||
        msg.includes('timed out') ||
        msg.includes('http 5') ||
        msg.includes('502') ||
        msg.includes('503') ||
        msg.includes('504')
    )
}

export function useIngressScanner(
    dirHandle: FileSystemDirectoryHandle | null,
    authorizedFetch: AuthorizedFetch,
    jobs: JobSummary[],
    uploadsEnabled: boolean = true,
    intervalMs: number = 5000
) {
    const { isLeader: isIngressLeader, tabId } = useTabLeader('windsurf:ingressScanner:lock')
    const [active, setActive] = React.useState(false)
    const [lastRunAt, setLastRunAt] = React.useState<number | null>(null)
    const [lastError, setLastError] = React.useState<string | null>(null)
    const [uploads, setUploads] = React.useState<IngressUploadItem[]>([])
    const inProgressRef = React.useRef<Set<string>>(new Set())
    const failedRef = React.useRef<Set<string>>(new Set())
    const retryAfterRef = React.useRef<Map<string, number>>(new Map())
    // Avoid hammering the server for the same checksum between scans while waiting for
    // Firestore job snapshots to catch up. Keep TTL short so deletions/re-association work.
    const recentlyHandledUntilRef = React.useRef<Map<string, number>>(new Map())
    // Track files that are still being copied so we only upload once they settle.
    const stabilityRef = React.useRef<Map<string, { size: number; mtimeMs: number; stableCount: number }>>(new Map())
    const [suspended, setSuspended] = React.useState(false)
    const { settings } = useSettings()

    const updateUpload = React.useCallback((id: string, partial: Partial<IngressUploadItem>) => {
        setUploads(prev => prev.map(u => (u.id === id ? { ...u, ...partial } : u)))
    }, [])

    const removeUpload = React.useCallback((id: string) => {
        setUploads(prev => prev.filter(u => u.id !== id))
    }, [])

    const timerRef = React.useRef<number | null>(null)
    const scanGenRef = React.useRef<number>(0)
    const scanInFlightRef = React.useRef<boolean>(false)
    const { refresh, loaded: fileIndexLoaded, scanStatus, snapshot } = useLocalFileIndex(dirHandle)
    const [remoteState, setRemoteState] = React.useState<IngressScannerSharedState | null>(null)

    const shouldStartNewUpload = React.useCallback(
        (sha: string) => {
            if (!uploadsEnabled) return false

            // Gold standard: if a job with this checksum already exists for this user, skip.
            if (jobs.some(j => j.sha256 === sha)) return false

            // If a job is already in "uploading" on the server, we only resume if we have a persisted job_id
            if (inProgressRef.current.has(sha)) return false

            const handledUntil = recentlyHandledUntilRef.current.get(sha) ?? 0
            if (Date.now() < handledUntil) return false

            if (failedRef.current.has(sha)) {
                const retryAfter = retryAfterRef.current.get(sha)
                if (!retryAfter || Date.now() < retryAfter) return false
                failedRef.current.delete(sha)
                retryAfterRef.current.delete(sha)
                if (failedRef.current.size === 0) setSuspended(false)
            }

            return true
        },
        [jobs, uploadsEnabled]
    )

    const uploadOne = React.useCallback(
        async (sha256: string, file: File, relativePath: string, existingJobId?: string) => {
            setUploads(prev => {
                assert(!prev.some(u => u.id === sha256), 'Upload already in progress')
                return [...prev, { id: sha256, relativePath, progress: 0, status: 'queued', error: null }]
            })

            try {
                inProgressRef.current.add(sha256)

                const result = await uploadVideoFile({
                    file,
                    quality: settings.uploadQuality,
                    authorizedFetch,
                    onProgress: percent => updateUpload(sha256, { progress: Math.round(percent * 100) }),
                    onStarted: () => updateUpload(sha256, { status: 'uploading' }),
                    sha256,
                    existingJobId,
                })
                const ttl = result === 'skipped' ? SKIPPED_TTL_MS : HANDLED_TTL_MS
                recentlyHandledUntilRef.current.set(sha256, Date.now() + ttl)

                if (result === 'skipped') {
                    removeUpload(sha256)
                    return
                }
                updateUpload(sha256, { progress: 100, status: 'done' })
                window.setTimeout(() => removeUpload(sha256), 3000)
            } catch (e: any) {
                console.error('Upload failed for', relativePath, e)
                const message = e?.message || String(e)
                setLastError(message)
                updateUpload(sha256, { status: 'error', error: message })
                failedRef.current.add(sha256)
                if (isTransientUploadError(e)) {
                    retryAfterRef.current.set(sha256, Date.now() + AUTO_RETRY_MS)
                } else {
                    retryAfterRef.current.delete(sha256)
                }
                setSuspended(true)
            } finally {
                inProgressRef.current.delete(sha256)
            }
        },
        [removeUpload, settings.uploadQuality, updateUpload, authorizedFetch]
    )

    const scanContinuously = React.useCallback(async () => {
        const myGen = scanGenRef.current
        if (!dirHandle) return

        // Prevent overlapping scans; without this it's easy to end up with multiple concurrent
        // loops scheduling timers with different captured props/sets.
        if (scanInFlightRef.current || !fileIndexLoaded) {
            if (scanGenRef.current !== myGen) return
            timerRef.current = window.setTimeout(scanContinuously, intervalMs)
            return
        }

        scanInFlightRef.current = true
        try {
            const result = await refresh()
            if (result) {
                const { snapshot, getFileForFingerprint } = result

                const stabilityByPath = new Map<string, { size: number; mtimeMs: number; stableCount: number }>()
                const REQUIRED_STABLE_SCANS = 1

                const isStable = (fingerprint: FileFingerprint) => {
                    const prev = stabilityRef.current.get(fingerprint.path)
                    let stableCount: number
                    if (prev && prev.size === fingerprint.size && prev.mtimeMs === fingerprint.mtimeMs) {
                        stableCount = prev.stableCount + 1
                    } else {
                        stableCount = 0
                    }
                    // Update stability tracking for this file
                    stabilityByPath.set(fingerprint.path, {
                        size: fingerprint.size,
                        mtimeMs: fingerprint.mtimeMs,
                        stableCount,
                    })
                    return stableCount >= REQUIRED_STABLE_SCANS
                }

                const work: Promise<void>[] = []

                const queueUpload = async (fingerprint: FileFingerprint) => {
                    if (!isStable(fingerprint)) return
                    const sha = fingerprint.sha256

                    const currentUploadingJob = jobs.find(j => j.sha256 === sha && j.status === 'uploading')
                    const shouldContinueUpload = !!currentUploadingJob && !inProgressRef.current.has(sha)

                    if (!shouldContinueUpload && !shouldStartNewUpload(sha)) return

                    const file = await getFileForFingerprint(fingerprint)
                    work.push(uploadOne(sha, file, fingerprint.path, currentUploadingJob?.id))
                }

                for (const fp of snapshot.fileFingerprints) {
                    // eslint-disable-next-line no-await-in-loop
                    await queueUpload(fp)
                }

                await Promise.all(work)

                // Update stability tracking for files that are still in the snapshot
                stabilityRef.current = stabilityByPath
            }
        } catch (e: any) {
            setLastError(e?.message || String(e))
        } finally {
            scanInFlightRef.current = false
        }

        setLastRunAt(Date.now())
        if (scanGenRef.current !== myGen) return
        timerRef.current = window.setTimeout(scanContinuously, intervalMs)
    }, [dirHandle, intervalMs, fileIndexLoaded, refresh, shouldStartNewUpload, uploadOne, jobs])

    React.useEffect(() => {
        if (timerRef.current) window.clearTimeout(timerRef.current)
        if (!dirHandle || !uploadsEnabled || !isIngressLeader) {
            setActive(false)
            return
        }
        setActive(true)
        scanGenRef.current += 1
        // immediate run, then interval
        scanContinuously()
        return () => {
            scanGenRef.current += 1
            if (timerRef.current) window.clearTimeout(timerRef.current)
            timerRef.current = null
        }
    }, [dirHandle, intervalMs, isIngressLeader, scanContinuously, uploadsEnabled])

    const retryFailed = React.useCallback(() => {
        if (!isIngressLeader) {
            requestIngressRetryFailed()
            return
        }
        failedRef.current = new Set()
        retryAfterRef.current = new Map()
        setLastError(null)
        setUploads(prev =>
            prev.map(u => (u.status === 'error' ? { ...u, status: 'queued', progress: 0, error: null } : u))
        )
        setSuspended(false) // This will trigger a new scan
    }, [isIngressLeader])

    const uploading = React.useMemo(
        () => uploads.filter(u => u.status === 'uploading' || u.status === 'queued').length,
        [uploads]
    )

    const detectedFiles = React.useMemo(() => snapshot?.fileFingerprints.length ?? 0, [snapshot])

    // Share ingress state cross-tab so non-leader tabs can show progress and results.
    React.useEffect(() => {
        if (!isIngressLeader) return
        const disposer = subscribeIngressScannerCommands(() => retryFailed())
        return disposer
    }, [isIngressLeader, retryFailed])

    React.useEffect(() => {
        if (isIngressLeader) return
        const disposer = subscribeIngressScannerState(next => setRemoteState(next))
        return disposer
    }, [isIngressLeader])

    React.useEffect(() => {
        if (!isIngressLeader) return

        let timeoutId: number | null = null
        timeoutId = window.setTimeout(() => {
            publishIngressScannerState({
                leaderTabId: tabId,
                active,
                lastRunAt,
                lastError,
                uploading,
                uploads,
                suspended,
                detectedFiles,
                scanStatus,
            })
        }, 200)

        return () => {
            if (timeoutId) window.clearTimeout(timeoutId)
        }
    }, [
        active,
        detectedFiles,
        isIngressLeader,
        lastError,
        lastRunAt,
        scanStatus,
        suspended,
        tabId,
        uploading,
        uploads,
    ])

    const effective = React.useMemo(() => {
        if (isIngressLeader) {
            return { active, lastRunAt, lastError, uploading, uploads, suspended, detectedFiles, scanStatus }
        }
        if (remoteState) {
            return {
                active: remoteState.active,
                lastRunAt: remoteState.lastRunAt,
                lastError: remoteState.lastError,
                uploading: remoteState.uploading,
                uploads: remoteState.uploads,
                suspended: remoteState.suspended,
                detectedFiles: remoteState.detectedFiles,
                scanStatus: remoteState.scanStatus,
            }
        }
        return { active: false, lastRunAt: null, lastError: null, uploading: 0, uploads: [], suspended: false, detectedFiles: 0, scanStatus: { phase: 'idle', total: 0, processed: 0 } }
    }, [active, detectedFiles, isIngressLeader, lastError, lastRunAt, remoteState, scanStatus, suspended, uploading, uploads])

    return {
        active: effective.active,
        isIngressLeader,
        lastRunAt: effective.lastRunAt,
        lastError: effective.lastError,
        uploading: effective.uploading,
        uploads: effective.uploads,
        suspended: effective.suspended,
        retryFailed,
        scanStatus: effective.scanStatus,
        detectedFiles: effective.detectedFiles,
    }
}
