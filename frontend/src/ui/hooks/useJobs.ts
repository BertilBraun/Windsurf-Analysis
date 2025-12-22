import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { JobDetail, JobSummary, ReportType } from '../types'
import { loadLastKnownPath, loadSetting, saveSetting, deleteSetting } from '../utils/idb'
import { assert } from '../utils/assert'
import { collection, doc, onSnapshot, query, where, type Unsubscribe } from 'firebase/firestore'
import { db } from '../../firebase'
import { useLocalFileIndex } from './useLocalFileIndex'

type UseJobsReturn = {
    jobs: JobSummary[]
    ready: boolean
    initialSyncComplete: boolean
    refreshJobDetail: (id: string) => Promise<JobDetail>
    deleteJob: (id: string) => Promise<void>
    reportJob: (id: string, type: ReportType, message: string) => Promise<void>
}

const _assertIsPercentage = (p: number) => {
    assert(0 <= p && p <= 1, `Percentage must be between 0 and 1: ${p}`)
}

export function useJobs(): UseJobsReturn {
    const { authorizedFetch, uid } = useAuth()
    const [jobs, setJobs] = React.useState<JobSummary[]>([])
    const [ready, setReady] = React.useState<boolean>(false)
    const [initialSyncComplete, setInitialSyncComplete] = React.useState<boolean>(false)
    const { shaToPaths } = useLocalFileIndex(null) // NOTE: automatically refreshed once the ingress scanner refreshes
    const readyRef = React.useRef<boolean>(false)
    const jobDetailCacheRef = React.useRef<Map<string, Promise<JobDetail>>>(new Map())
    const realtimeUnsubRef = React.useRef<Unsubscribe | null>(null)
    const jobUnsubsRef = React.useRef<Map<string, Unsubscribe>>(new Map())
    const jobsByIdRef = React.useRef<Map<string, JobSummary>>(new Map())
    const activeJobIdsRef = React.useRef<Set<string>>(new Set())
    const hydratedJobIdsRef = React.useRef<Set<string>>(new Set())
    const shaToPathsRef = React.useRef<Map<string, string[]>>(new Map())

    const publishJobsFromRef = React.useCallback(() => {
        const next = Array.from(jobsByIdRef.current.values()).sort((a, b) => {
            const ta = Date.parse(a.updated_at || a.created_at || '') || 0
            const tb = Date.parse(b.updated_at || b.created_at || '') || 0
            return tb - ta
        })
        setJobs(next)
    }, [])

    const getLocalPathsForSha = React.useCallback((sha: string | null | undefined): string[] => {
        if (!sha) return []
        const paths = shaToPathsRef.current.get(sha) ?? []
        return paths.slice()
    }, [])

    const getLastKnownPathForSha = React.useCallback(async (sha: string | null | undefined): Promise<string | null> => {
        if (!sha) return null
        return await loadLastKnownPath(sha)
    }, [])

    // When the local file snapshot updates, refresh any affected jobs in a single batch.
    React.useEffect(() => {
        shaToPathsRef.current = shaToPaths

        void (async () => {
            let changed = false
            for (const [jobId, job] of jobsByIdRef.current.entries()) {
                const jobSha = job.original_checksum_sha256
                const jobPath = job.local_relative_path
                if (!jobSha) continue
                const paths = getLocalPathsForSha(jobSha)
                const preferred = paths.length ? paths[0] : null
                const keep = jobPath && paths.some(p => p === jobPath) ? jobPath : preferred
                const prevPaths = job.local_relative_paths || []
                const nextLastKnown = paths.length === 0 ? await getLastKnownPathForSha(jobSha) : null
                const same =
                    jobPath === keep &&
                    JSON.stringify(prevPaths) === JSON.stringify(paths) &&
                    (job.last_known_local_path || null) === (nextLastKnown || null)
                if (same) continue
                jobsByIdRef.current.set(jobId, {
                    ...job,
                    local_relative_path: keep,
                    local_relative_paths: paths,
                    last_known_local_path: nextLastKnown,
                })
                jobDetailCacheRef.current.delete(jobId)
                changed = true
            }
            if (changed) publishJobsFromRef()
        })()
    }, [getLastKnownPathForSha, publishJobsFromRef, shaToPaths])

    const stopRealtime = React.useCallback(() => {
        if (realtimeUnsubRef.current) {
            realtimeUnsubRef.current()
            realtimeUnsubRef.current = null
        }
        for (const unsub of jobUnsubsRef.current.values()) unsub()
        jobUnsubsRef.current.clear()
        jobsByIdRef.current.clear()
        activeJobIdsRef.current.clear()
        hydratedJobIdsRef.current.clear()
    }, [])

    const recomputeInitialSyncComplete = React.useCallback(() => {
        if (!readyRef.current) {
            setInitialSyncComplete(false)
            return
        }
        const active = activeJobIdsRef.current
        if (active.size === 0) {
            setInitialSyncComplete(true)
            return
        }
        // IMPORTANT: only count jobs as "synced" once we've hydrated them into jobsByIdRef
        // (includes checksum + derived local paths). Otherwise ingress can start with an incomplete
        // knownChecksums set and re-create/re-upload jobs on reload.
        let hydratedActiveCount = 0
        for (const id of active) if (hydratedJobIdsRef.current.has(id)) hydratedActiveCount += 1
        setInitialSyncComplete(hydratedActiveCount >= active.size)
    }, [])

    // Start realtime subscriptions on mount; cleanup on unmount.
    React.useEffect(() => {
        stopRealtime()
        setJobs([])
        setReady(false)
        readyRef.current = false
        setInitialSyncComplete(false)

        if (!uid) return

        const userJobsQ = query(
            collection(db, 'user_jobs'),
            where('user_id', '==', uid),
            where('deleted_at', '==', null)
        )

        realtimeUnsubRef.current = onSnapshot(
            userJobsQ,
            snap => {
                setReady(true)
                readyRef.current = true
                const activeJobIds = new Set<string>()
                for (const d of snap.docs) {
                    const data = d.data() as any
                    const jobId = String(data?.job_id ?? '')
                    if (jobId) activeJobIds.add(jobId)
                }
                activeJobIdsRef.current = activeJobIds
                for (const existingId of Array.from(hydratedJobIdsRef.current)) {
                    if (!activeJobIds.has(existingId)) hydratedJobIdsRef.current.delete(existingId)
                }

                // Remove jobs that are no longer associated to the user (regardless of whether we still have a listener).
                for (const existingId of Array.from(jobsByIdRef.current.keys())) {
                    if (activeJobIds.has(existingId)) continue
                    jobUnsubsRef.current.get(existingId)?.()
                    jobUnsubsRef.current.delete(existingId)
                    jobsByIdRef.current.delete(existingId)
                    jobDetailCacheRef.current.delete(existingId)
                    void deleteSetting(`jobDetail:${existingId}`)
                }

                // Subscribe to newly-added jobs
                for (const jobId of activeJobIds) {
                    if (jobUnsubsRef.current.has(jobId)) continue
                    // If we already have the job in memory (e.g. was terminal and we unsubscribed earlier),
                    // don't re-subscribe.
                    if (jobsByIdRef.current.has(jobId)) continue

                    const jobDocRef = doc(db, 'jobs', jobId)
                    let unsub: Unsubscribe | null = null
                    unsub = onSnapshot(
                        jobDocRef,
                        jobSnap => {
                            if (!jobSnap.exists()) return
                            const j = jobSnap.data() as any

                            const toIso = (v: any): string => {
                                if (!v) return new Date(0).toISOString()
                                if (typeof v === 'string') return v
                                if (typeof v?.toDate === 'function') return v.toDate().toISOString()
                                if (v instanceof Date) return v.toISOString()
                                return String(v)
                            }

                            const summaryBase: JobSummary = {
                                id: String(j?.job_id ?? jobId),
                                status: String(j?.status ?? 'pending') as any,
                                created_at: toIso(j?.created_at),
                                updated_at: toIso(j?.updated_at),
                                original_checksum_sha256: String(j?.original_checksum_sha256 ?? ''),
                                dominant_orientation: Number(j?.dominant_orientation ?? 0),
                            }

                            void (async () => {
                                const sha = summaryBase.original_checksum_sha256
                                const local_relative_paths = getLocalPathsForSha(sha)
                                const local_relative_path = local_relative_paths.length ? local_relative_paths[0] : null
                                const last_known_local_path =
                                    local_relative_paths.length === 0 ? await getLastKnownPathForSha(sha) : null
                                const summary: JobSummary = {
                                    ...summaryBase,
                                    local_relative_path,
                                    local_relative_paths,
                                    last_known_local_path,
                                }
                                jobsByIdRef.current.set(jobId, summary)
                                hydratedJobIdsRef.current.add(jobId)
                                publishJobsFromRef()
                                recomputeInitialSyncComplete()

                                // Optimization: stop listening once job is terminal.
                                const s = String(summaryBase.status || '').toLowerCase()
                                const isTerminal =
                                    s === 'succeeded' || s === 'failed' || s === 'canceled' || s === 'cancelled'
                                if (isTerminal && unsub) {
                                    unsub()
                                    jobUnsubsRef.current.delete(jobId)
                                }
                            })()
                        },
                        err => {
                            console.warn('jobs doc snapshot error', err)
                        }
                    )
                    jobUnsubsRef.current.set(jobId, unsub)
                }

                // If there are no active jobs, initial sync is complete immediately.
                recomputeInitialSyncComplete()
            },
            err => {
                console.warn('user_jobs snapshot error; falling back to one-time fetch', err)
                setReady(true)
                readyRef.current = true
                setInitialSyncComplete(true)
                stopRealtime()
            }
        )

        return () => stopRealtime()
    }, [uid, stopRealtime, recomputeInitialSyncComplete])

    const refreshJobDetail = React.useCallback(
        async (id: string): Promise<JobDetail> => {
            const cache = jobDetailCacheRef.current
            const inFlight = cache.get(id)
            if (inFlight) return inFlight

            const key = `jobDetail:${id}`
            const fetchPromise = (async () => {
                // 1) Try persistent cache first
                const persisted = await loadSetting<JobDetail>(key)
                if (persisted) {
                    const local_relative_paths = getLocalPathsForSha(persisted.original_checksum_sha256)
                    const local_relative_path = local_relative_paths.length ? local_relative_paths[0] : null
                    const last_known_local_path =
                        local_relative_paths.length === 0
                            ? await getLastKnownPathForSha(persisted.original_checksum_sha256)
                            : null
                    return { ...persisted, local_relative_path, local_relative_paths, last_known_local_path }
                }

                // 2) Fetch from network, validate, persist, and return
                const res = await authorizedFetch(`/jobs/${id}`)
                const data = (await res.json()) as JobDetail
                assert(data.status === 'succeeded')

                for (const track of data.tracks) {
                    _assertIsPercentage(track.start_percent)
                    _assertIsPercentage(track.end_percent)
                    for (const detection of track.detections) {
                        _assertIsPercentage(detection.time_percent)
                        for (const b of detection.bbox) {
                            _assertIsPercentage(b)
                        }
                    }
                }
                for (const transform of data.stabilization_transforms) {
                    _assertIsPercentage(transform.time_percent)
                    assert(Number.isFinite(transform.dx))
                    assert(Number.isFinite(transform.dy))
                    assert(Number.isFinite(transform.da))
                }

                // Persist the canonical server response (without derived path)
                await saveSetting(key, data)

                const local_relative_paths = getLocalPathsForSha(data.original_checksum_sha256)
                const local_relative_path = local_relative_paths.length ? local_relative_paths[0] : null
                const last_known_local_path =
                    local_relative_paths.length === 0
                        ? await getLastKnownPathForSha(data.original_checksum_sha256)
                        : null
                return { ...data, local_relative_path, local_relative_paths, last_known_local_path }
            })()

            cache.set(id, fetchPromise)

            try {
                return await fetchPromise
            } catch (err) {
                // Remove failed promise from cache to allow retry
                cache.delete(id)
                throw err
            }
        },
        [authorizedFetch, getLastKnownPathForSha, getLocalPathsForSha]
    )

    const deleteJob = React.useCallback(
        async (id: string) => {
            await authorizedFetch(`/jobs/${id}`, { method: 'DELETE' })
            setJobs(jobs.filter(job => job.id !== id))
            jobDetailCacheRef.current.delete(id)
            await deleteSetting(`jobDetail:${id}`)
        },
        [authorizedFetch, jobs, setJobs]
    )

    const reportJob = React.useCallback(
        async (id: string, type: ReportType, message: string) => {
            await authorizedFetch(`/jobs/${id}/report`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type, message }),
            })
        },
        [authorizedFetch]
    )

    React.useEffect(
        () => () => {
            stopRealtime()
        },
        [stopRealtime]
    )

    return { jobs, ready, initialSyncComplete, refreshJobDetail, deleteJob, reportJob }
}
