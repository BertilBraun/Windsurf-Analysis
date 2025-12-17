import React from 'react'
import { AuthProvider } from './auth/AuthProvider'
import { Router } from './routes/Router'
import { initAnalytics, installClickTracking } from './utils/analytics'

export const App: React.FC = () => {
    React.useEffect(() => {
        initAnalytics()
        installClickTracking()
    }, [])

    return (
        <AuthProvider>
            <div style={{ fontFamily: 'Inter, system-ui, Arial', lineHeight: 1.4 }}>
                <Router />
            </div>
        </AuthProvider>
    )
}
