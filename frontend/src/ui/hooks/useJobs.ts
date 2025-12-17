import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { JobDetail, JobSummary, ReportType } from '../types'
import { getPathForSha, loadSetting, saveSetting, deleteSetting } from '../utils/idb'
import { assert } from '../utils/assert'
import { collection, doc, onSnapshot, query, where, type Unsubscribe } from 'firebase/firestore'
import { db } from '../../firebase'

type UseJobsReturn = {
    jobs: JobSummary[]
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
    const jobDetailCacheRef = React.useRef<Map<string, Promise<JobDetail>>>(new Map())
    const realtimeUnsubRef = React.useRef<Unsubscribe | null>(null)
    const jobUnsubsRef = React.useRef<Map<string, Unsubscribe>>(new Map())
    const jobsByIdRef = React.useRef<Map<string, JobSummary>>(new Map())

    const stopRealtime = React.useCallback(() => {
        if (realtimeUnsubRef.current) {
            realtimeUnsubRef.current()
            realtimeUnsubRef.current = null
        }
        for (const unsub of jobUnsubsRef.current.values()) unsub()
        jobUnsubsRef.current.clear()
        jobsByIdRef.current.clear()
    }, [])

    const tick = React.useCallback(async () => {
        const res = await authorizedFetch('/jobs')
        const data = (await res.json()) as { jobs: JobSummary[] }
        // Enrich with local_relative_path from IDB mapping
        const enriched: JobSummary[] = await Promise.all(
            data.jobs.map(async job => ({
                ...job,
                local_relative_path: await getPathForSha(job.original_checksum_sha256),
            }))
        )
        setJobs(enriched)
        const anyOpen = enriched.some(
            job =>
                job.status === 'pending' ||
                job.status === 'starting' ||
                job.status === 'orientation' ||
                job.status === 'stabilization' ||
                job.status === 'detection' ||
                job.status === 'appearance' ||
                job.status === 'tracking'
        )
        // Note: realtime mode handles stopping naturally; this is just for initial hydration UX.
        if (!anyOpen) return
    }, [authorizedFetch])

    // Start realtime subscriptions on mount; cleanup on unmount.
    React.useEffect(() => {
        stopRealtime()
        setJobs([])

        if (!uid) return

        const userJobsQ = query(
            collection(db, 'user_jobs'),
            where('user_id', '==', uid),
            where('deleted_at', '==', null)
        )

        realtimeUnsubRef.current = onSnapshot(
            userJobsQ,
            snap => {
                const activeJobIds = new Set<string>()
                for (const d of snap.docs) {
                    const data = d.data() as any
                    const jobId = String(data?.job_id ?? '')
                    if (jobId) activeJobIds.add(jobId)
                }

                // Unsubscribe removed jobs
                for (const existingId of Array.from(jobUnsubsRef.current.keys())) {
                    if (!activeJobIds.has(existingId)) {
                        jobUnsubsRef.current.get(existingId)?.()
                        jobUnsubsRef.current.delete(existingId)
                        jobsByIdRef.current.delete(existingId)
                        jobDetailCacheRef.current.delete(existingId)
                        void deleteSetting(`jobDetail:${existingId}`)
                    }
                }

                // Subscribe to newly-added jobs
                for (const jobId of activeJobIds) {
                    if (jobUnsubsRef.current.has(jobId)) continue

                    const jobDocRef = doc(db, 'jobs', jobId)
                    const unsub = onSnapshot(
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
                                const local_relative_path = summaryBase.original_checksum_sha256
                                    ? await getPathForSha(summaryBase.original_checksum_sha256)
                                    : null
                                const summary: JobSummary = { ...summaryBase, local_relative_path }
                                jobsByIdRef.current.set(jobId, summary)

                                const next = Array.from(jobsByIdRef.current.values()).sort((a, b) => {
                                    const ta = Date.parse(a.updated_at || a.created_at || '') || 0
                                    const tb = Date.parse(b.updated_at || b.created_at || '') || 0
                                    return tb - ta
                                })
                                setJobs(next)
                            })()
                        },
                        err => {
                            console.warn('jobs doc snapshot error', err)
                        }
                    )
                    jobUnsubsRef.current.set(jobId, unsub)
                }
            },
            err => {
                console.warn('user_jobs snapshot error; falling back to one-time fetch', err)
                stopRealtime()
                tick()
            }
        )

        // Kick an initial backend fetch to populate immediately (nice UX) until first snapshots arrive.
        void tick()

        return () => stopRealtime()
    }, [uid, stopRealtime, tick])

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
                    const local_relative_path = await getPathForSha(persisted.original_checksum_sha256)
                    return { ...persisted, local_relative_path }
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

                const local_relative_path = await getPathForSha(data.original_checksum_sha256)
                return { ...data, local_relative_path }
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
        [authorizedFetch]
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

    return { jobs, refreshJobDetail, deleteJob, reportJob }
}
