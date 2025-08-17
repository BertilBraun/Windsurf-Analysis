import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { JobDetail, JobStatus, JobSummary, ReportType, Track, TrackDetection } from '../types'

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
    if (p < 0 || p > 1) throw new Error(`Percentage must be between 0 and 1: ${p}`)
}

export function useJobs(): UseJobsReturn {
    const { authorizedFetch } = useAuth()
    const [jobs, setJobs] = React.useState<JobSummary[]>([])
    const [isPolling, setIsPolling] = React.useState(false)
    const pollingRef = React.useRef<number | null>(null)

    const tick = React.useCallback(async () => {
        const res = await authorizedFetch('/jobs')
        const data = (await res.json()) as { jobs: JobSummary[] }
        setJobs(data.jobs)
        const anyOpen = data.jobs.some(j => j.status === 'pending' || j.status === 'running')
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
        async (id: string) => {
            // TODO cache?
            const res = await authorizedFetch(`/jobs/${id}`)
            const data = (await res.json()) as JobDetail
            if (data.tracks) {
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
            }
            return data
        },
        [authorizedFetch]
    )

    const deleteJob = React.useCallback(
        async (id: string) => {
            await authorizedFetch(`/jobs/${id}`, { method: 'DELETE' })
            setJobs(jobs.filter(j => j.id !== id))
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
