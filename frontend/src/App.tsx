import {
    createUserWithEmailAndPassword,
    onAuthStateChanged,
    sendEmailVerification,
    sendPasswordResetEmail,
    signInWithEmailAndPassword,
    signInWithPopup,
    signOut,
    type User,
} from 'firebase/auth'
import { doc, getDoc, onSnapshot, serverTimestamp, setDoc, type Unsubscribe } from 'firebase/firestore'
import { useEffect, useMemo, useState } from 'react'
import { callBackend } from './api'
import { auth, db, googleProvider } from './firebase'

type WhoAmI = {
    uid?: string
    email?: string
    email_verified?: boolean
    name?: string
    picture?: string
    issuer?: string
}

export default function App() {
    const [user, setUser] = useState<User | null>(null)
    const [out, setOut] = useState<string>('')
    const [email, setEmail] = useState<string>('')
    const [password, setPassword] = useState<string>('')
    const [password2, setPassword2] = useState<string>('')
    const [authMode, setAuthMode] = useState<'none' | 'signin' | 'signup'>('none')
    const [subUnsub, setSubUnsub] = useState<Unsubscribe | null>(null)
    const signedIn = !!user
    const emailNeedsVerification = !!user?.email && !user.emailVerified
    const canUseApp = signedIn && !emailNeedsVerification

    useEffect(() => onAuthStateChanged(auth, setUser), [])

    const authLabel = useMemo(() => {
        if (!user) return 'Signed out'
        const base = `Signed in: ${user.email ?? user.uid}`
        if (emailNeedsVerification) return `${base} (email not verified)`
        return base
    }, [user])

    async function doGoogleSignIn() {
        try {
            const result = await signInWithPopup(auth, googleProvider)
            setOut(JSON.stringify({ signedInAs: result.user.email ?? result.user.uid }, null, 2))
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    async function doEmailSignUp() {
        try {
            if (password !== password2) throw new Error('Passwords do not match.')
            const result = await createUserWithEmailAndPassword(auth, email.trim(), password)
            await sendEmailVerification(result.user)
            setOut(
                JSON.stringify(
                    {
                        signedUp: result.user.email ?? result.user.uid,
                        next: 'Please verify your email. (Check inbox/spam)',
                    },
                    null,
                    2
                )
            )
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    async function doEmailSignIn() {
        try {
            const result = await signInWithEmailAndPassword(auth, email.trim(), password)
            if (!result.user.emailVerified) {
                setOut(
                    JSON.stringify(
                        {
                            signedIn: result.user.email ?? result.user.uid,
                            warning: 'Email not verified yet. Please verify to continue.',
                        },
                        null,
                        2
                    )
                )
            } else {
                setOut(JSON.stringify({ signedInAs: result.user.email ?? result.user.uid }, null, 2))
            }
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    async function doResetPassword() {
        try {
            const targetEmail = email.trim()
            if (!targetEmail) throw new Error('Enter your email first.')
            await sendPasswordResetEmail(auth, targetEmail)
            setOut(JSON.stringify({ sent: `Password reset email sent to ${targetEmail}.` }, null, 2))
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    async function doResendVerification() {
        try {
            if (!auth.currentUser) throw new Error('Not signed in.')
            await sendEmailVerification(auth.currentUser)
            setOut(JSON.stringify({ sent: 'Verification email sent. Check inbox/spam.' }, null, 2))
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    async function doCheckVerified() {
        try {
            if (!auth.currentUser) throw new Error('Not signed in.')
            await auth.currentUser.reload()
            // Force-refresh token so backend sees updated email_verified claim.
            await auth.currentUser.getIdToken(true)
            setUser(auth.currentUser)
            setOut(JSON.stringify({ emailVerified: auth.currentUser.emailVerified }, null, 2))
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    async function doSignOut() {
        try {
            if (subUnsub) {
                subUnsub()
                setSubUnsub(null)
            }
            await signOut(auth)
            setOut('Signed out.')
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    async function doWhoAmI() {
        try {
            const data = await callBackend<WhoAmI>('/whoami')
            setOut(JSON.stringify(data, null, 2))
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    async function doBackendFirestorePing() {
        try {
            const data = await callBackend('/firestore/ping', { method: 'POST' })
            setOut(JSON.stringify(data, null, 2))
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    async function doFrontendFirestorePing() {
        try {
            if (!user) throw new Error('Not signed in.')
            const ref = doc(db, 'frontendPings', user.uid)
            await setDoc(ref, { ts: serverTimestamp() }, { merge: true })
            const snap = await getDoc(ref)
            setOut(JSON.stringify({ wroteAndRead: snap.data() ?? null }, null, 2))
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    function startSub() {
        try {
            if (!canUseApp) throw new Error('Sign in with a verified email first.')
            if (subUnsub) return
            const ref = doc(db, 'test', 'test')
            const unsub = onSnapshot(
                ref,
                snap => {
                    setOut(
                        JSON.stringify(
                            { subscription: 'update', path: 'test/test', data: snap.data() ?? null },
                            null,
                            2
                        )
                    )
                },
                err => {
                    setOut(JSON.stringify({ subscription: 'error', error: String(err) }, null, 2))
                }
            )
            setSubUnsub(() => unsub)
            setOut(JSON.stringify({ subscription: 'started', path: 'test/test' }, null, 2))
        } catch (e) {
            setOut(JSON.stringify({ error: String(e) }, null, 2))
        }
    }

    function stopSub() {
        if (subUnsub) {
            subUnsub()
            setSubUnsub(null)
            setOut(JSON.stringify({ subscription: 'stopped' }, null, 2))
        }
    }

    // Auto-refresh verification status while user is unverified.
    useEffect(() => {
        if (!emailNeedsVerification) return
        const t = window.setInterval(() => {
            void doCheckVerified()
        }, 5000)
        return () => window.clearInterval(t)
    }, [emailNeedsVerification])

    useEffect(() => {
        return () => {
            if (subUnsub) subUnsub()
        }
    }, [subUnsub])

    return (
        <div className="page">
            <div className="card">
                <h1>Windsurf Analysis</h1>
                <p className="muted">React + TypeScript (Vite) on Firebase Hosting</p>

                <p>{authLabel}</p>

                {!signedIn && (
                    <>
                        <div className="row">
                            <button
                                onClick={() => {
                                    setAuthMode('signup')
                                    setOut('')
                                }}
                            >
                                Show signup
                            </button>
                            <button
                                onClick={() => {
                                    setAuthMode('signin')
                                    setOut('')
                                }}
                            >
                                Show email signin
                            </button>
                            <button onClick={doGoogleSignIn}>Sign in (Google)</button>
                        </div>

                        {authMode !== 'none' && (
                            <div style={{ marginTop: 12 }}>
                                <div className="row">
                                    <input
                                        value={email}
                                        onChange={e => setEmail(e.target.value)}
                                        placeholder="email"
                                        autoComplete="email"
                                    />
                                    <input
                                        value={password}
                                        onChange={e => setPassword(e.target.value)}
                                        placeholder="password"
                                        type="password"
                                        autoComplete={authMode === 'signin' ? 'current-password' : 'new-password'}
                                    />
                                    {authMode === 'signup' && (
                                        <input
                                            value={password2}
                                            onChange={e => setPassword2(e.target.value)}
                                            placeholder="repeat password"
                                            type="password"
                                            autoComplete="new-password"
                                        />
                                    )}
                                </div>

                                <div className="row">
                                    {authMode === 'signin' ? (
                                        <>
                                            <button onClick={doEmailSignIn} disabled={!email || !password}>
                                                Sign in
                                            </button>
                                            <button onClick={doResetPassword} disabled={!email}>
                                                Forgot password
                                            </button>
                                        </>
                                    ) : (
                                        <button onClick={doEmailSignUp} disabled={!email || !password || !password2}>
                                            Sign up
                                        </button>
                                    )}
                                    <button
                                        onClick={() => {
                                            setAuthMode('none')
                                            setPassword('')
                                            setPassword2('')
                                        }}
                                    >
                                        Cancel
                                    </button>
                                </div>
                            </div>
                        )}
                    </>
                )}

                <div className="row">
                    <button onClick={doSignOut} disabled={!signedIn}>
                        Sign out
                    </button>
                </div>

                {emailNeedsVerification && (
                    <div style={{ marginTop: 12 }}>
                        <p>
                            Please verify your email ({user?.email}). After verifying, this page will auto-refresh the
                            status (or click “I verified”).
                        </p>
                        <div className="row">
                            <button onClick={doResendVerification}>Resend verification email</button>
                            <button onClick={doCheckVerified}>I verified</button>
                        </div>
                    </div>
                )}

                <div className="row">
                    <button onClick={doWhoAmI} disabled={!canUseApp}>
                        Call backend: /whoami
                    </button>
                    <button onClick={doBackendFirestorePing} disabled={!canUseApp}>
                        Backend Firestore: /firestore/ping
                    </button>
                    <button onClick={doFrontendFirestorePing} disabled={!canUseApp}>
                        Frontend Firestore: write/read
                    </button>
                    <button onClick={startSub} disabled={!canUseApp || !!subUnsub}>
                        Start subscription test/test
                    </button>
                    <button onClick={stopSub} disabled={!subUnsub}>
                        Stop subscription
                    </button>
                </div>

                <pre className="out">{out}</pre>
            </div>
        </div>
    )
}
