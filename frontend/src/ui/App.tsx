/**
 * @fileoverview Main entry point for the React application UI.
 * Sets up global providers, analytics, and the root routing structure.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { useLocation } from 'react-router-dom'
import { AuthProvider } from './auth/AuthProvider'
import { Router } from './routes/Router'
import { initAnalytics, installClickTracking, isAnalyticsEnabled } from './utils/analytics'
import { AnalyticsConsentBanner } from './components/AnalyticsConsentBanner'
import { UnsupportedBrowserBanner } from './components/UnsupportedBrowserBanner'

/**
 * The root component of the application.
 * Handles global side effects like analytics initialization and language synchronization,
 * and provides the authentication context to the component tree.
 */
export const App: React.FC = () => {
    const { i18n } = useTranslation()
    const location = useLocation()

    React.useEffect(() => {
        if (isAnalyticsEnabled()) {
            initAnalytics()
            installClickTracking()
        }
    }, [])

    React.useEffect(() => {
        document.documentElement.lang = i18n.language
    }, [i18n.language])

    return (
        <AuthProvider>
            <div style={{ fontFamily: 'Inter, system-ui, Arial', lineHeight: 1.4 }}>
                {location.pathname === '/analyzer' && <UnsupportedBrowserBanner />}
                <Router />
                <AnalyticsConsentBanner />
            </div>
        </AuthProvider>
    )
}
