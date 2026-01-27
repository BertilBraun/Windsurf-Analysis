/**
 * @module Router
 * @description Main routing configuration for the application.
 *
 * Orchestrates the application's navigation structure, including public marketing
 * pages, protected analyzer routes, and ephemeral demo sessions. Includes
 * automated page view tracking for analytics.
 */

import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthProvider'
import { HomePage } from '../screens/HomePage'
import { FaqPage } from '../screens/FaqPage'
import { AnalyzerPage } from '../screens/AnalyzerPage'
import { PricingPage } from '../screens/PricingPage'
import { LegalPage } from '../screens/LegalPage'
import { TechnicalPage } from '../screens/TechnicalPage'
import { AppShellLayout } from '../components/AppShell'
import { LoginPage } from '../screens/LoginPage'
import { SignupPage } from '../screens/SignupPage'
import { DemoPage } from '../screens/DemoPage'
import { LogoButton } from '../components/LogoButton'
import { Button } from '../components/Button'
import { ConsentModal } from '../components/ConsentModal'
import { trackPageView } from '../utils/analytics'
import { auth } from '../../firebase'
import { createUserWithEmailAndPassword, inMemoryPersistence, setPersistence, signInAnonymously } from 'firebase/auth'

const DemoRoute: React.FC = () => {
    const { t } = useTranslation()
    const { isAuthReady, isSignedIn, authorizedFetch, uid } = useAuth()
    const navigate = useNavigate()
    const [authError, setAuthError] = React.useState<string | null>(null)

    React.useEffect(() => {
        if (!isAuthReady) return
        if (isSignedIn) return

        let cancelled = false
            ; (async () => {
                setAuthError(null)
                try {
                    // Demo sessions should be ephemeral: refresh wipes the session.
                    await setPersistence(auth, inMemoryPersistence)
                    await signInAnonymously(auth)
                } catch (e: any) {
                    if (cancelled) return
                    setAuthError(e?.message || String(e))
                }
            })()

        return () => {
            cancelled = true
        }
    }, [isAuthReady, isSignedIn])

    React.useEffect(() => {
        if (!isAuthReady) return
        if (!isSignedIn) return
        if (!uid) return

        let cancelled = false

            ; (async () => {
                try {
                    // Demo does not require Terms/Privacy modal; just ensure backend user doc exists
                    // so /jobs creation doesn't 404 for new anonymous sessions.
                    const createRes = await authorizedFetch(`/users/${uid}`, { method: 'POST' })
                    if (!createRes.ok && createRes.status !== 400) throw new Error(await createRes.text())
                } catch (e) {
                    if (cancelled) return
                    const message = e instanceof Error ? e.message : String(e)
                    console.warn('Failed to ensure demo user record', e)
                    setAuthError(message)
                }
            })()

        return () => {
            cancelled = true
        }
    }, [authorizedFetch, isAuthReady, isSignedIn, uid])

    if (!isAuthReady || !isSignedIn) {
        return (
            <div className="min-h-dvh bg-white text-slate-900">
                <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                    <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                        <LogoButton onClick={() => navigate('/')} />
                        <div className="flex-1" />
                    </div>
                </header>
                <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-10 space-y-2">
                    <div className="text-sm text-slate-600">{t('routes.analyzer.loadingSession')}</div>
                    {authError && <div className="text-sm text-red-700 break-words">{authError}</div>}
                </main>
            </div>
        )
    }

    return <DemoPage onGoHome={() => navigate('/')} />
}

const AnalyzerRoute: React.FC = () => {
    const { t } = useTranslation()
    const {
        user,
        isAuthReady,
        isAuthenticated,
        isSignedIn,
        needsEmailVerification,
        email,
        logout,
        resendVerificationEmail,
        refreshVerificationStatus,
        authorizedFetch,
        uid,
    } = useAuth()
    const navigate = useNavigate()
    const location = useLocation()
    const [authMode, setAuthMode] = React.useState<'login' | 'signup'>('login')
    const [clearingDemoSession, setClearingDemoSession] = React.useState(false)
    const [consentState, setConsentState] = React.useState<
        'checking' | 'required' | 'ready' | { kind: 'error'; message: string }
    >('checking')
    const [consentSubmitting, setConsentSubmitting] = React.useState(false)
    const [consentReloadKey, setConsentReloadKey] = React.useState(0)

    React.useEffect(() => {
        if (!isAuthReady) return
        if (!isSignedIn) return
        if (!user) return

        // NOTE/TODO: Demo uses anonymous or generated demo accounts. Those should not be carried into
        // the full analyzer (it would trigger email verification). Sign out and show login/signup instead.
        const isAnon = !!(user as any)?.isAnonymous
        const emailStr = String(user.email || '')
        const isGeneratedDemoEmail = emailStr.startsWith('demo+') && emailStr.endsWith('@example.com')
        if (!isAnon && !isGeneratedDemoEmail) return

        setClearingDemoSession(true)
        logout()
    }, [isAuthReady, isSignedIn, logout, user])

    React.useEffect(() => {
        if (!clearingDemoSession) return
        if (isSignedIn) return
        setClearingDemoSession(false)
    }, [clearingDemoSession, isSignedIn])

    React.useEffect(() => {
        if (!isAuthReady) return
        if (!isSignedIn) return
        if (!isAuthenticated) return
        if (needsEmailVerification) return
        if (!uid) return

        let cancelled = false
        const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms))

            ; (async () => {
                setConsentState('checking')
                for (let attempt = 0; attempt < 3; attempt++) {
                    try {
                        const res = await authorizedFetch(`/users/${uid}`)

                        if (res.status === 404) {
                            const createRes = await authorizedFetch(`/users/${uid}`, { method: 'POST' })
                            if (!createRes.ok) throw new Error(await createRes.text())
                            // Retry GET after creating the user record.
                            continue
                        }

                        if (!res.ok) throw new Error(await res.text())
                        const data = (await res.json()) as {
                            terms_accepted_at?: string | null
                            privacy_accepted_at?: string | null
                        }
                        if (cancelled) return
                        const needsConsent = !data?.terms_accepted_at || !data?.privacy_accepted_at
                        setConsentState(needsConsent ? 'required' : 'ready')
                        return
                    } catch (e) {
                        if (attempt < 2) {
                            await sleep(250 * Math.pow(2, attempt))
                            continue
                        }
                        if (cancelled) return
                        const message = e instanceof Error ? e.message : String(e)
                        console.warn('Failed to load user consent state', e)
                        setConsentState({ kind: 'error', message })
                    }
                }
            })()

        return () => {
            cancelled = true
        }
    }, [authorizedFetch, isAuthenticated, isAuthReady, isSignedIn, needsEmailVerification, uid, consentReloadKey])

    if (!isAuthReady) {
        // Avoid a flash of the logged-out UI while Firebase restores the persisted session.
        return (
            <div className="min-h-dvh bg-white text-slate-900">
                <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                    <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                        <LogoButton onClick={() => navigate('/')} />
                        <div className="flex-1" />
                    </div>
                </header>
                <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-10">
                    <div className="text-sm text-slate-600">{t('routes.analyzer.loadingSession')}</div>
                </main>
            </div>
        )
    }

    if (clearingDemoSession) {
        return (
            <div className="min-h-dvh bg-white text-slate-900">
                <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                    <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                        <LogoButton onClick={() => navigate('/')} />
                        <div className="flex-1" />
                    </div>
                </header>
                <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-10">
                    <div className="text-sm text-slate-600">{t('routes.analyzer.loadingSession')}</div>
                </main>
            </div>
        )
    }

    if (!isSignedIn) {
        return (
            <div className="min-h-dvh bg-gradient-to-br from-brand-50 via-white to-white text-slate-900">
                <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                    <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                        <LogoButton onClick={() => navigate('/')} />
                        <div className="flex-1" />
                    </div>
                </header>

                <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-10">
                    <div className="relative mx-auto w-full max-w-md">
                        <div className="pointer-events-none absolute inset-x-0 -top-12 mx-auto h-56 w-56 rounded-full bg-brand-50 blur-3xl opacity-70" />
                        {authMode === 'signup' ? (
                            <SignupPage
                                onBackToLogin={() => setAuthMode('login')}
                                onSuccess={() => {
                                    // After signup the user will be signed in; if email is unverified,
                                    // this route will immediately show the verification screen.
                                }}
                            />
                        ) : (
                            <LoginPage
                                onSignup={() => setAuthMode('signup')}
                                onSuccess={() => {
                                    // Stay on /analyzer; the authenticated state will re-render the Analyzer view.
                                    // Preserve deep-link intent if any (future-proof).
                                    const from = (location.state as any)?.from
                                    if (typeof from === 'string' && from.startsWith('/')) navigate(from)
                                }}
                            />
                        )}
                    </div>
                </main>
            </div>
        )
    }

    if (needsEmailVerification) {
        return (
            <div className="min-h-dvh bg-white text-slate-900">
                <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                    <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                        <LogoButton onClick={() => navigate('/')} />
                        <div className="flex-1" />
                    </div>
                </header>

                <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-10">
                    <div className="max-w-xl">
                        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                            <h3 className="m-0">{t('routes.analyzer.verify.title')}</h3>
                            <p className="text-sm text-slate-600 mt-2">
                                <Trans
                                    i18nKey="routes.analyzer.verify.body"
                                    components={{ b: <b /> }}
                                    values={{ email: email ?? t('routes.analyzer.verify.yourEmail') }}
                                />
                            </p>
                            <div className="flex flex-wrap gap-2 mt-4">
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="md"
                                    onClick={() => void resendVerificationEmail()}
                                    text={t('routes.analyzer.verify.resend')}
                                />
                                <Button
                                    type="button"
                                    variant="secondary"
                                    size="md"
                                    onClick={() => void refreshVerificationStatus()}
                                    text={t('routes.analyzer.verify.verified')}
                                />
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="md"
                                    onClick={logout}
                                    text={t('routes.analyzer.verify.signOut')}
                                />
                            </div>
                        </div>
                    </div>
                </main>
            </div>
        )
    }

    if (!isAuthenticated) {
        // Should be unreachable (isAuthenticated === isSignedIn && verified),
        // but keep a safe fallback.
        return <Navigate to="/analyzer" replace />
    }

    if (consentState === 'checking') {
        return (
            <div className="min-h-dvh bg-white text-slate-900">
                <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                    <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                        <LogoButton onClick={() => navigate('/')} />
                        <div className="flex-1" />
                    </div>
                </header>
                <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-10">
                    <div className="text-sm text-slate-600">{t('routes.analyzer.loadingSession')}</div>
                </main>
            </div>
        )
    }

    if (typeof consentState === 'object' && consentState.kind === 'error') {
        return (
            <div className="min-h-dvh bg-white text-slate-900">
                <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                    <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                        <LogoButton onClick={() => navigate('/')} />
                        <div className="flex-1" />
                    </div>
                </header>
                <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-10">
                    <div className="max-w-xl rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-3">
                        <h3 className="m-0">{t('routes.analyzer.loadingSession')}</h3>
                        <div className="text-sm text-slate-600 break-words">{consentState.message}</div>
                        <div className="flex gap-2">
                            <Button
                                type="button"
                                variant="secondary"
                                size="md"
                                onClick={() => setConsentReloadKey(v => v + 1)}
                                text={t('common.actions.retry')}
                            />
                            <Button
                                type="button"
                                variant="outline"
                                size="md"
                                onClick={logout}
                                text={t('routes.analyzer.verify.signOut')}
                            />
                        </div>
                    </div>
                </main>
            </div>
        )
    }

    if (consentState === 'required') {
        return (
            <div className="min-h-dvh bg-white text-slate-900">
                <ConsentModal
                    isSubmitting={consentSubmitting}
                    onSubmit={async marketingConsent => {
                        if (!uid) return
                        setConsentSubmitting(true)
                        try {
                            // Ensure the backend user doc exists with required fields before patching consent.
                            const createRes = await authorizedFetch(`/users/${uid}`, { method: 'POST' })
                            if (!createRes.ok) throw new Error(await createRes.text())

                            const res = await authorizedFetch(`/users/${uid}/consent`, {
                                method: 'PATCH',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    terms_accepted: true,
                                    marketing_consent: marketingConsent,
                                }),
                            })
                            if (!res.ok) throw new Error(await res.text())
                            setConsentState('ready')
                        } catch (e) {
                            console.error('Failed to update consent', e)
                            const message = e instanceof Error ? e.message : String(e)
                            setConsentState({ kind: 'error', message })
                        } finally {
                            setConsentSubmitting(false)
                        }
                    }}
                />
            </div>
        )
    }

    return <AnalyzerPage onGoHome={() => navigate('/')} onGoPricing={() => navigate('/pricing')} />
}

/**
 * The primary application router component.
 *
 * Defines all top-level routes, manages page view analytics tracking, and
 * handles conditional rendering for authenticated vs. public views.
 *
 * @returns {JSX.Element} The rendered route configuration.
 */
export const Router: React.FC = () => {
    const location = useLocation()

    React.useEffect(() => {
        trackPageView(`${location.pathname}${location.search}${location.hash}`)
    }, [location.pathname, location.search, location.hash])

    return (
        <Routes>
            <Route path="/demo" element={<DemoRoute />} />
            <Route
                path="/analyzer"
                element={
                    <AnalyzerRoute />
                }
            />

            {/* Marketing / public pages use real URL routes and do not require login. */}
            <Route element={<AppShellLayout />}>
                <Route index element={<HomePage />} />
                <Route path="pricing" element={<PricingPage />} />
                <Route path="faq" element={<FaqPage />} />
                <Route path="technical" element={<TechnicalPage />} />
                <Route path="terms" element={<LegalPage kind="terms" />} />
                <Route path="privacy" element={<LegalPage kind="privacy" />} />
                <Route path="impressum" element={<LegalPage kind="impressum" />} />
                <Route path="contact" element={<LegalPage kind="contact" />} />
            </Route>

            {/* Back-compat: if something links to /login or /signup, show login inside Analyzer. */}
            <Route path="/login" element={<Navigate to="/analyzer" replace />} />
            <Route path="/signup" element={<Navigate to="/analyzer" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
    )
}
