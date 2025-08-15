import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { JobDetail, JobStatus, JobSummary } from '../types'

type UseJobsReturn = {
    jobs: JobSummary[]
    isPolling: boolean
    startPolling: () => void
    stopPolling: () => void
    refreshJobDetail: (id: string) => Promise<JobDetail>
    setJobs: React.Dispatch<React.SetStateAction<JobSummary[]>>
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

    const refreshJobDetail = React.useCallback(async (id: string) => {
        const res = await authorizedFetch(`/jobs/${id}`)
        const data = (await res.json()) as JobDetail
        return data
    }, [authorizedFetch])

    React.useEffect(() => () => {
        if (pollingRef.current) window.clearInterval(pollingRef.current)
    }, [])

    return { jobs, isPolling, startPolling, stopPolling, refreshJobDetail, setJobs }
}


