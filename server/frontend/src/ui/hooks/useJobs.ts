import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { JobDetail, JobSummary, ReportType } from '../types'
import { getPathForSha } from '../utils/idb'
import { assert } from '../utils/assert'

type UseJobsReturn = {
    jobs: JobSummary[]
    isPolling: boolean
    startPolling: () => void
    stopPolling: () => void
    refreshJobDetail: (id: string) => Promise<JobDetail>
    deleteJob: (id: string) => Promise<void>
    reportJob: (id: string, type: ReportType, message: string) => Promise<void>
}

const _assertIsPercentage = (p: number) => {
    assert(0 <= p && p <= 1, `Percentage must be between 0 and 1: ${p}`)
}

export function useJobs(): UseJobsReturn {
    const { authorizedFetch } = useAuth()
    const [jobs, setJobs] = React.useState<JobSummary[]>([])
    const [isPolling, setIsPolling] = React.useState(false)
    const pollingRef = React.useRef<number | null>(null)

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
                job.status === 'stabilization' ||
                job.status === 'detection' ||
                job.status === 'appearance' ||
                job.status === 'tracking'
        )
        if (!anyOpen) {
            if (pollingRef.current) window.clearInterval(pollingRef.current)
            pollingRef.current = null
            setIsPolling(false)
        }
    }, [authorizedFetch])

    const startPolling = React.useCallback(() => {
        if (pollingRef.current) window.clearInterval(pollingRef.current)
        pollingRef.current = window.setInterval(tick, 10000)
        setIsPolling(true)
        // Do an immediate refresh
        tick()
    }, [tick])

    const stopPolling = React.useCallback(() => {
        if (pollingRef.current) window.clearInterval(pollingRef.current)
        pollingRef.current = null
        setIsPolling(false)
    }, [])

    const refreshJobDetail = React.useCallback(
        async (id: string): Promise<JobDetail> => {
            const res = await authorizedFetch(`/jobs/${id}`)
            const data = (await res.json()) as JobDetail
            assert(data.status === 'succeeded')

            // TODO cache? If the job is done, we could cache the job detail
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
            const local_relative_path = await getPathForSha(data.original_checksum_sha256)
            return { ...data, local_relative_path }
        },
        [authorizedFetch]
    )

    const deleteJob = React.useCallback(
        async (id: string) => {
            await authorizedFetch(`/jobs/${id}`, { method: 'DELETE' })
            setJobs(jobs.filter(job => job.id !== id))
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
            if (pollingRef.current) window.clearInterval(pollingRef.current)
        },
        []
    )

    return { jobs, isPolling, startPolling, stopPolling, refreshJobDetail, deleteJob, reportJob }
}
