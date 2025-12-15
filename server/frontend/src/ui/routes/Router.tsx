import React from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth/AuthProvider'
import { HomePage } from '../screens/HomePage'
import { FaqPage } from '../screens/FaqPage'
import { AnalyzerPage } from '../screens/AnalyzerPage'
import { PricingPage } from '../screens/PricingPage'
import { LegalPage } from '../screens/LegalPage'
import { ImpressumPage } from '../screens/ImpressumPage'
import { AppShellLayout } from '../components/AppShell'
import { SingleInstanceGuard } from '../components/SingleInstanceGuard'
import { LoginPage } from '../screens/LoginPage'
import { LogoButton } from '../components/LogoButton'

const AnalyzerRoute: React.FC = () => {
    const { isAuthenticated } = useAuth()
    const navigate = useNavigate()
    const location = useLocation()

    if (!isAuthenticated) {
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
                        <div className="text-sm text-slate-600 mb-4">Log in to access the Analyzer.</div>
                        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                            <LoginPage
                                onSignup={() => navigate('/pricing')}
                                onSuccess={() => {
                                    // Stay on /analyzer; the authenticated state will re-render the Analyzer view.
                                    // Preserve deep-link intent if any (future-proof).
                                    const from = (location.state as any)?.from
                                    if (typeof from === 'string' && from.startsWith('/')) navigate(from)
                                }}
                            />
                        </div>
                    </div>
                </main>
            </div>
        )
    }

    return <AnalyzerPage onGoHome={() => navigate('/')} onGoPricing={() => navigate('/pricing')} />
}

export const Router: React.FC = () => {
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
                <Route path="impressum" element={<ImpressumPage />} />
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
