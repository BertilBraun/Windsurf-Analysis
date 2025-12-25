import React from 'react'
import { useTranslation } from 'react-i18next'
import { AuthProvider } from './auth/AuthProvider'
import { Router } from './routes/Router'
import { initAnalytics, installClickTracking, isAnalyticsEnabled } from './utils/analytics'
import { AnalyticsConsentBanner } from './components/AnalyticsConsentBanner'
import { UnsupportedBrowserBanner } from './components/UnsupportedBrowserBanner'

export const App: React.FC = () => {
    const { i18n } = useTranslation()

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
                <UnsupportedBrowserBanner />
                <Router />
                <AnalyticsConsentBanner />
            </div>
        </AuthProvider>
    )
}
