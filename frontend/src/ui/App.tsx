import React from 'react'
import { AuthProvider } from './auth/AuthProvider'
import { Router } from './routes/Router'
import { initAnalytics, installClickTracking, isAnalyticsEnabled } from './utils/analytics'
import { AnalyticsConsentBanner } from './components/AnalyticsConsentBanner'

export const App: React.FC = () => {
    React.useEffect(() => {
        if (isAnalyticsEnabled()) {
            initAnalytics()
            installClickTracking()
        }
    }, [])

    return (
        <AuthProvider>
            <div style={{ fontFamily: 'Inter, system-ui, Arial', lineHeight: 1.4 }}>
                <Router />
                <AnalyticsConsentBanner />
            </div>
        </AuthProvider>
    )
}
