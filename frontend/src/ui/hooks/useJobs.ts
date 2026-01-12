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
    deleteJobs: (ids: string[]) => Promise<number>
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
    const userJobsUnsubRef = React.useRef<Unsubscribe | null>(null)
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

    const getPathsForSha = React.useCallback(
        async (
            sha: string
        ): Promise<{
            local_relative_paths: string[]
            local_relative_path: string | null
            last_known_local_path: string | null
        }> => {
            const local_relative_paths = [...(shaToPathsRef.current.get(sha) || [])]
            const local_relative_path = local_relative_paths.length ? local_relative_paths[0] : null
            const last_known_local_path = local_relative_paths.length === 0 ? await loadLastKnownPath(sha) : null
            return { local_relative_paths, local_relative_path, last_known_local_path }
        },
        []
    )

    // When the local file snapshot updates, refresh any affected jobs in a single batch.
    React.useEffect(() => {
        shaToPathsRef.current = shaToPaths

        void (async () => {
            let changed = false
            for (const [jobId, job] of jobsByIdRef.current.entries()) {
                const { local_relative_paths, local_relative_path, last_known_local_path } = await getPathsForSha(
                    job.sha256
                )

                const pathEqual = job.local_relative_path === local_relative_path
                const pathsEqual = JSON.stringify(job.local_relative_paths) === JSON.stringify(local_relative_paths)
                const lastKnownPathEqual = job.last_known_local_path === last_known_local_path
                if (pathEqual && pathsEqual && lastKnownPathEqual) continue

                jobsByIdRef.current.set(jobId, {
                    ...job,
                    local_relative_path,
                    local_relative_paths,
                    last_known_local_path,
                })
                changed = true
            }
            if (changed) publishJobsFromRef()
        })()
    }, [publishJobsFromRef, shaToPaths])

    const stopRealtime = React.useCallback(() => {
        if (userJobsUnsubRef.current) {
            userJobsUnsubRef.current()
            userJobsUnsubRef.current = null
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

    const onJobSnapshot = React.useCallback(
        async (jobId: string, jobData: any, unsub: Unsubscribe | null) => {
            const toIso = (v: any): string => {
                if (!v) return new Date(0).toISOString()
                if (typeof v === 'string') return v
                if (typeof v?.toDate === 'function') return v.toDate().toISOString()
                if (v instanceof Date) return v.toISOString()
                return String(v)
            }

            assert(jobData?.status !== undefined, 'Job status is required')
            assert(jobData?.created_at !== undefined, 'Job created_at is required')
            assert(jobData?.updated_at !== undefined, 'Job updated_at is required')
            assert(jobData?.dominant_orientation !== undefined, 'Job dominant_orientation is required')

            if (!activeJobIdsRef.current.has(jobId)) return

            const sha = jobId

            const summary: JobSummary = {
                id: jobId,
                status: jobData.status,
                created_at: toIso(jobData.created_at),
                updated_at: toIso(jobData.updated_at),
                sha256: sha,
                dominant_orientation: jobData.dominant_orientation,
                ...(await getPathsForSha(sha)),
            }
            jobsByIdRef.current.set(jobId, summary)
            hydratedJobIdsRef.current.add(jobId)
            publishJobsFromRef()
            recomputeInitialSyncComplete()

            // Stop listening once job is terminal (we only need realtime while it's processing).
            const status = summary.status
            const isTerminal = status === 'succeeded' || status === 'failed' || status === 'canceled'
            if (isTerminal) {
                assert(unsub !== null, 'Unsubscribe function is required')
                unsub?.()
                jobUnsubsRef.current.delete(jobId)
            }
        },
        [activeJobIdsRef, hydratedJobIdsRef, publishJobsFromRef, recomputeInitialSyncComplete, getPathsForSha]
    )
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

        userJobsUnsubRef.current = onSnapshot(
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
                let removedAny = false
                for (const existingId of Array.from(jobsByIdRef.current.keys())) {
                    if (activeJobIds.has(existingId)) continue
                    jobUnsubsRef.current.get(existingId)?.()
                    jobUnsubsRef.current.delete(existingId)
                    jobsByIdRef.current.delete(existingId)
                    jobDetailCacheRef.current.delete(existingId)
                    void deleteSetting(`jobDetail:${existingId}`)
                    removedAny = true
                }
                if (removedAny) publishJobsFromRef()

                // Subscribe to newly-added jobs. Unsubscribe per-job listeners once jobs become terminal.
                for (const jobId of activeJobIds) {
                    if (jobUnsubsRef.current.has(jobId)) continue
                    // If we already have the job in memory (e.g. it was terminal and we unsubscribed earlier),
                    // don't re-subscribe.
                    if (jobsByIdRef.current.has(jobId)) continue

                    let unsub: Unsubscribe | null = null
                    unsub = onSnapshot(
                        doc(db, 'jobs', jobId),
                        jobSnap => {
                            if (!jobSnap.exists()) return
                            const jobData = jobSnap.data() as any
                            onJobSnapshot(jobId, jobData, unsub)
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
                    const { local_relative_paths, local_relative_path, last_known_local_path } = await getPathsForSha(
                        persisted.sha256
                    )
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

                return { ...data, ...(await getPathsForSha(data.sha256)) }
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
        [authorizedFetch, getPathsForSha]
    )

    const deleteJobs = React.useCallback(
        async (ids: string[]) => {
            const job_ids = Array.from(new Set(ids.filter(Boolean)))
            if (job_ids.length === 0) return 0

            const res = await authorizedFetch(`/jobs/bulk-delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_ids }),
            })
            if (!res.ok) throw new Error(await res.text())
            const data = (await res.json()) as { deleted?: number }

            setJobs(prev => prev.filter(job => !job_ids.includes(job.id)))
            for (const id of job_ids) {
                jobDetailCacheRef.current.delete(id)
                await deleteSetting(`jobDetail:${id}`)
            }
            return Number(data.deleted || 0)
        },
        [authorizedFetch]
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

    return { jobs, ready, initialSyncComplete, refreshJobDetail, deleteJobs, reportJob }
}
