import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'

type JobStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'canceled'

type JobSummary = {
    id: string
    video_id: string
    model: string
    status: JobStatus
    created_at: string
    updated_at: string
}

type JobDetail = JobSummary & {
    results_json?: Record<string, unknown> | null
}

type ReportType = 'missed_detection' | 'false_association' | 'other'

const apiBase = '/api/v1'

export const App: React.FC = () => {
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [authHeader, setAuthHeader] = useState<string | null>(null)

    const [jobs, setJobs] = useState<JobSummary[]>([])
    const [selectedJob, setSelectedJob] = useState<JobDetail | null>(null)
    const [isPolling, setIsPolling] = useState(false)
    const pollingRef = useRef<number | null>(null)

    const makeAuthHeader = useCallback((e: string, p: string) => 'Basic ' + btoa(`${e}:${p}`), [])

    const authorizedFetch = useCallback(
        async (input: RequestInfo, init?: RequestInit) => {
            if (!authHeader) throw new Error('Not authenticated')
            console.log('fetching', input, init)
            const res = await fetch(input, {
                ...init,
                headers: {
                    ...(init?.headers || {}),
                    Authorization: authHeader,
                },
            })
            console.log('res', res)
            if (!res.ok) throw new Error(await res.text())
            return res
        },
        [authHeader]
    )

    const startPolling = useCallback(() => {
        if (pollingRef.current) window.clearInterval(pollingRef.current)
        const tick = async () => {
            try {
                const res = await authorizedFetch(`${apiBase}/jobs`)
                const data = (await res.json()) as { jobs: JobSummary[] }
                setJobs(data.jobs)
                const anyOpen = data.jobs.some(j => j.status === 'pending' || j.status === 'running')
                const interval = anyOpen ? 10000 : 60000 // 10s or 60s
                // TODO? Does it even need to poll if no jobs are open?
                if (pollingRef.current) window.clearInterval(pollingRef.current)
                pollingRef.current = window.setInterval(tick, interval)
            } catch (e) {
                // stop polling on error
                if (pollingRef.current) window.clearInterval(pollingRef.current)
            }
        }
        tick()
        setIsPolling(true)
    }, [authorizedFetch])

    const stopPolling = useCallback(() => {
        if (pollingRef.current) window.clearInterval(pollingRef.current)
        pollingRef.current = null
        setIsPolling(false)
    }, [])

    const onLogin = useCallback(() => {
        setAuthHeader(makeAuthHeader(email, password))
    }, [email, password, makeAuthHeader])

    const onSelectFileAndUpload = useCallback(async () => {
        if (!authHeader) return
        const picker = document.createElement('input')
        picker.type = 'file'
        picker.accept = 'video/*'
        picker.onchange = async () => {
            if (!picker.files || picker.files.length === 0) return
            const file = picker.files[0]
            // Compute checksum of original bytes in browser
            const arrayBuffer = await file.arrayBuffer()
            const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer)
            const hashArray = Array.from(new Uint8Array(hashBuffer))
            const sha256 = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')

            // TODO: Quick check if the checksum is already in the database
            // TODO: If it is, show a message and don't upload

            const form = new FormData()
            form.append('file', new Blob([arrayBuffer], { type: file.type }), file.name)
            form.append('original_file_path', file.name)
            form.append('original_checksum_sha256', sha256)
            form.append('yolo_model', 'windsurfing/2025_08_09_100epochs.pt')
            form.append('reid_model', 'common/osnet_ain_x1_0_msmt17.pth')

            await authorizedFetch(`${apiBase}/jobs/upload`, { method: 'POST', body: form })
            startPolling()
        }
        picker.click()
    }, [authHeader, authorizedFetch, startPolling])

    const refreshJobDetail = useCallback(
        async (id: string) => {
            const res = await authorizedFetch(`${apiBase}/jobs/${id}`)
            const data = (await res.json()) as JobDetail
            setSelectedJob(data)
        },
        [authorizedFetch]
    )

    const onDeleteJob = useCallback(
        async (id: string) => {
            await authorizedFetch(`${apiBase}/jobs/${id}`, { method: 'DELETE' })
            startPolling()
        },
        [authorizedFetch, startPolling]
    )

    const onReportJob = useCallback(
        async (id: string, type: ReportType, message: string) => {
            await authorizedFetch(`${apiBase}/jobs/${id}/report`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type, message }),
            })
        },
        [authorizedFetch]
    )

    useEffect(() => () => stopPolling(), [stopPolling])

    return (
        <div style={{ fontFamily: 'Inter, system-ui, Arial', margin: '24px', lineHeight: 1.4 }}>
            <h2>Windsurf Analysis – Test Console</h2>

            <section style={{ marginBottom: 24 }}>
                <h3>Auth</h3>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <input placeholder="email" value={email} onChange={e => setEmail(e.target.value)} />
                    <input placeholder="password" type="password" value={password} onChange={e => setPassword(e.target.value)} />
                    <button onClick={onLogin} disabled={!email || !password}>Use credentials</button>
                    <button onClick={isPolling ? stopPolling : startPolling} disabled={!authHeader}>
                        {isPolling ? 'Stop polling' : 'Start polling'}
                    </button>
                </div>
            </section>

            <section style={{ marginBottom: 24 }}>
                <h3>Jobs</h3>
                <div style={{ marginBottom: 12 }}>
                    <button onClick={onSelectFileAndUpload} disabled={!authHeader}>Submit new job</button>
                </div>
                <div>
                    {(jobs || []).map(j => (
                        <div key={j.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: 8, borderBottom: '1px solid #eee' }}>
                            <span style={{ width: 100, fontFamily: 'monospace' }}>{j.id.slice(0, 8)}</span>
                            <StatusBadge status={j.status} />
                            <span style={{ flex: 1 }}>{j.model}</span>
                            <button onClick={() => refreshJobDetail(j.id)} disabled={!authHeader}>Open</button>
                            <button onClick={() => onDeleteJob(j.id)} disabled={!authHeader}>Delete</button>
                            <ReportDropdown onSubmit={(type, message) => onReportJob(j.id, type, message)} disabled={!authHeader} />
                        </div>
                    ))}
                </div>
            </section>

            {selectedJob && (
                <section>
                    <h3>Job Detail</h3>
                    <div style={{ padding: 12, border: '1px solid #eee', borderRadius: 8 }}>
                        <div style={{ marginBottom: 8 }}>
                            <StatusBadge status={selectedJob.status} />
                            <span style={{ marginLeft: 8 }}>Job {selectedJob.id}</span>
                        </div>
                        <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(selectedJob.results_json ?? {}, null, 2)}</pre>
                    </div>
                </section>
            )}
        </div>
    )
}

const StatusBadge: React.FC<{ status: JobStatus }> = ({ status }) => {
    const color =
        status === 'succeeded' ? '#10b981' :
            status === 'failed' ? '#ef4444' :
                status === 'running' ? '#3b82f6' :
                    status === 'pending' ? '#f59e0b' : '#9ca3af'
    return (
        <span style={{ background: color, color: 'white', borderRadius: 12, padding: '2px 8px', fontSize: 12 }}>{status}</span>
    )
}

const ReportDropdown: React.FC<{ onSubmit: (t: ReportType, m: string) => void; disabled?: boolean }> = ({ onSubmit, disabled }) => {
    const [type, setType] = useState<ReportType>('missed_detection')
    const [message, setMessage] = useState('')
    return (
        <div style={{ display: 'inline-flex', gap: 6 }}>
            <select value={type} onChange={e => setType(e.target.value as ReportType)} disabled={disabled}>
                <option value="missed_detection">Missed detection</option>
                <option value="false_association">False association</option>
                <option value="other">Other</option>
            </select>
            <input placeholder="message" value={message} onChange={e => setMessage(e.target.value)} disabled={disabled} />
            <button onClick={() => onSubmit(type, message)} disabled={disabled || !message}>Report</button>
        </div>
    )
}


