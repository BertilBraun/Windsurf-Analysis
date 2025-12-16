import { useCallback, useEffect, useRef, useState } from 'react'
import { callBackend } from '../../api'
import type { JobDetail, JobSummary } from '../../types'

export function useJobs() {
    const [jobs, setJobs] = useState<JobSummary[]>([])
    const [isPolling, setIsPolling] = useState(false)
    const timerRef = useRef<number | null>(null)

    const refresh = useCallback(async () => {
        const data = await callBackend<{ jobs: JobSummary[] }>('/jobs')
        setJobs(data.jobs)
        const anyOpen = data.jobs.some(j =>
            ['pending', 'starting', 'orientation', 'stabilization', 'detection', 'appearance', 'tracking'].includes(
                j.status
            )
        )
        if (!anyOpen) {
            if (timerRef.current) window.clearInterval(timerRef.current)
            timerRef.current = null
            setIsPolling(false)
        }
    }, [])

    const startPolling = useCallback(() => {
        if (timerRef.current) window.clearInterval(timerRef.current)
        timerRef.current = window.setInterval(() => {
            void refresh()
        }, 10_000)
        setIsPolling(true)
        void refresh()
    }, [refresh])

    const stopPolling = useCallback(() => {
        if (timerRef.current) window.clearInterval(timerRef.current)
        timerRef.current = null
        setIsPolling(false)
    }, [])

    useEffect(
        () => () => {
            if (timerRef.current) window.clearInterval(timerRef.current)
        },
        []
    )

    const getJobDetail = useCallback(async (id: string) => {
        return await callBackend<JobDetail>(`/jobs/${id}`)
    }, [])

    const deleteJob = useCallback(async (id: string) => {
        await callBackend(`/jobs/${id}`, { method: 'DELETE' })
        setJobs(prev => prev.filter(j => j.id !== id))
    }, [])

    return { jobs, isPolling, refresh, startPolling, stopPolling, getJobDetail, deleteJob }
}

