import { useEffect, useMemo, useState } from 'react'
import type { User } from 'firebase/auth'
import { onAuthStateChanged, signInWithEmailAndPassword, signOut } from 'firebase/auth'
import { auth } from '../firebase'
import { callBackend } from '../api'
import { uploadToModal, computeSha256 } from './utils/uploader'
import { useJobs } from './hooks/useJobs'
import type { JobDetail } from '../types'

export function Analyzer() {
    const [user, setUser] = useState<User | null>(auth.currentUser)
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [busy, setBusy] = useState(false)
    const [out, setOut] = useState<string>('')
    const [file, setFile] = useState<File | null>(null)
    const [progress, setProgress] = useState<number>(0)
    const [selected, setSelected] = useState<JobDetail | null>(null)

    const canUseApp = !!user && !!user.emailVerified

    useEffect(() => onAuthStateChanged(auth, setUser), [])

    const { jobs, startPolling, stopPolling, refresh, getJobDetail, deleteJob } = useJobs()

    useEffect(() => {
        if (canUseApp) startPolling()
        return () => stopPolling()
    }, [canUseApp, startPolling, stopPolling])

    const authLabel = useMemo(() => {
        if (!user) return 'Signed out'
        const base = `Signed in: ${user.email ?? user.uid}`
        if (!user.emailVerified) return `${base} (email not verified)`
        return base
    }, [user])

    async function doLogin() {
        setBusy(true)
        setOut('')
        try {
            await signInWithEmailAndPassword(auth, email.trim(), password)
            setOut('Signed in.')
        } catch (e) {
            setOut(String(e))
        } finally {
            setBusy(false)
        }
    }

    async function doLogout() {
        await signOut(auth)
        setSelected(null)
    }

    async function doUpload() {
        if (!file) return
        setBusy(true)
        setOut('')
        setProgress(0)
        try {
            const sha256 = await computeSha256(file)
            const created = await callBackend<{ job_id: string; status: string }>('/jobs', {
                method: 'POST',
                body: { original_checksum_sha256: sha256 },
            })
            if (created.status !== 'pending') {
                setOut(`Duplicate detected, job status=${created.status}. Skipping upload.`)
                await refresh()
                return
            }
            await uploadToModal(created.job_id, file, p => setProgress(p))
            setOut(`Uploaded. Job: ${created.job_id}`)
            await refresh()
            startPolling()
        } catch (e: any) {
            setOut(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    async function openJob(id: string) {
        setBusy(true)
        setOut('')
        try {
            const d = await getJobDetail(id)
            setSelected(d)
        } catch (e: any) {
            setOut(String(e?.message || e))
        } finally {
            setBusy(false)
        }
    }

    return (
        <div className="page">
            <div className="card">
                <h1>Windsurf Analysis</h1>
                <p className="muted">Firebase Hosting frontend → Cloud Run (jobs) + Modal (uploads)</p>

                <p>{authLabel}</p>

                {!user && (
                    <>
                        <div className="row">
                            <input value={email} onChange={e => setEmail(e.target.value)} placeholder="email" />
                            <input
                                value={password}
                                onChange={e => setPassword(e.target.value)}
                                placeholder="password"
                                type="password"
                            />
                            <button onClick={doLogin} disabled={!email || !password || busy}>
                                Sign in
                            </button>
                        </div>
                    </>
                )}

                {user && (
                    <div className="row">
                        <button onClick={doLogout} disabled={busy}>
                            Sign out
                        </button>
                        <button onClick={() => refresh()} disabled={!canUseApp || busy}>
                            Refresh jobs
                        </button>
                    </div>
                )}

                {user && !user.emailVerified && <p className="muted">Verify your email to use the service.</p>}

                {canUseApp && (
                    <>
                        <hr />
                        <h3>Upload</h3>
                        <div className="row">
                            <input type="file" accept="video/*" onChange={e => setFile(e.target.files?.[0] ?? null)} />
                            <button onClick={doUpload} disabled={!file || busy}>
                                Upload
                            </button>
                            <span className="muted">{Math.round(progress * 100)}%</span>
                        </div>

                        <hr />
                        <h3>Jobs</h3>
                        {jobs.length === 0 ? (
                            <p className="muted">No jobs yet.</p>
                        ) : (
                            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                {jobs.map(j => (
                                    <div
                                        key={j.id}
                                        style={{
                                            display: 'flex',
                                            gap: 8,
                                            alignItems: 'center',
                                            justifyContent: 'space-between',
                                        }}
                                    >
                                        <div style={{ flex: 1 }}>
                                            <div>
                                                <b>{j.status}</b> — {j.original_checksum_sha256.slice(0, 12)}…
                                            </div>
                                            <div className="muted" style={{ fontSize: 12 }}>
                                                {j.id}
                                            </div>
                                        </div>
                                        <div style={{ display: 'flex', gap: 8 }}>
                                            <button onClick={() => openJob(j.id)} disabled={busy}>
                                                Open
                                            </button>
                                            <button onClick={() => deleteJob(j.id)} disabled={busy}>
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}

                        {selected && (
                            <>
                                <hr />
                                <h3>Job detail</h3>
                                <pre className="out">{JSON.stringify(selected, null, 2)}</pre>
                            </>
                        )}
                    </>
                )}

                {out && <pre className="out">{out}</pre>}
            </div>
        </div>
    )
}

