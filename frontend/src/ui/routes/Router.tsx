import React from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthProvider'
import { HomePage } from '../screens/HomePage'
import { FaqPage } from '../screens/FaqPage'
import { AnalyzerPage } from '../screens/AnalyzerPage'
import { PricingPage } from '../screens/PricingPage'
import { LegalPage } from '../screens/LegalPage'
import { AppShellLayout } from '../components/AppShell'
import { SingleInstanceGuard } from '../components/SingleInstanceGuard'
import { LoginPage } from '../screens/LoginPage'
import { SignupPage } from '../screens/SignupPage'
import { LogoButton } from '../components/LogoButton'
import { trackPageView } from '../utils/analytics'

const AnalyzerRoute: React.FC = () => {
    const {
        isAuthReady,
        isAuthenticated,
        isSignedIn,
        needsEmailVerification,
        email,
        logout,
        resendVerificationEmail,
        refreshVerificationStatus,
    } = useAuth()
    const navigate = useNavigate()
    const location = useLocation()
    const [authMode, setAuthMode] = React.useState<'login' | 'signup'>('login')

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
                    <div className="text-sm text-slate-600">Loading your session…</div>
                </main>
            </div>
        )
    }

    if (!isSignedIn) {
        return (
            <div className="min-h-dvh bg-white text-slate-900">
                <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                    <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                        <LogoButton onClick={() => navigate('/')} />
                        <div className="flex-1" />
                    </div>
                </header>

                <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-10">
                    <div className="max-w-md">
                        <div className="text-sm text-slate-600 mb-4">
                            {authMode === 'signup'
                                ? 'Create an account to access the Analyzer.'
                                : 'Log in to access the Analyzer.'}
                        </div>
                        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
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
                            <h3 className="m-0">Verify your email to continue</h3>
                            <p className="text-sm text-slate-600 mt-2">
                                The backend requires a verified email address. Please verify{' '}
                                <b>{email ?? 'your email'}</b> (check inbox/spam), then click “I verified”.
                            </p>
                            <div className="flex flex-wrap gap-2 mt-4">
                                <button
                                    type="button"
                                    onClick={() => void resendVerificationEmail()}
                                    className="rounded-lg border border-slate-200 px-3 py-2 text-sm"
                                >
                                    Resend verification email
                                </button>
                                <button
                                    type="button"
                                    onClick={() => void refreshVerificationStatus()}
                                    className="rounded-lg bg-slate-900 text-white px-3 py-2 text-sm"
                                >
                                    I verified
                                </button>
                                <button
                                    type="button"
                                    onClick={logout}
                                    className="rounded-lg border border-slate-200 px-3 py-2 text-sm"
                                >
                                    Sign out
                                </button>
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

    return <AnalyzerPage onGoHome={() => navigate('/')} onGoPricing={() => navigate('/pricing')} />
}

export const Router: React.FC = () => {
    const location = useLocation()

    React.useEffect(() => {
        trackPageView(`${location.pathname}${location.search}${location.hash}`)
    }, [location.pathname, location.search, location.hash])

    return (
        <Routes>
            <Route
                path="/analyzer"
                element={
                    <SingleInstanceGuard>
                        <AnalyzerRoute />
                    </SingleInstanceGuard>
                }
            />

            {/* Marketing / public pages use real URL routes and do not require login. */}
            <Route element={<AppShellLayout />}>
                <Route index element={<HomePage />} />
                <Route path="pricing" element={<PricingPage />} />
                <Route path="faq" element={<FaqPage />} />
                <Route path="terms" element={<LegalPage kind="terms" />} />
                <Route path="privacy" element={<LegalPage kind="privacy" />} />
                <Route path="contact" element={<LegalPage kind="contact" />} />
            </Route>

            {/* Back-compat: if something links to /login or /signup, show login inside Analyzer. */}
            <Route path="/login" element={<Navigate to="/analyzer" replace />} />
            <Route path="/signup" element={<Navigate to="/analyzer" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
    )
}
