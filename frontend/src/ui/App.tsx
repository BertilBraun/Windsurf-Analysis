import React from 'react'
import { AuthProvider } from './auth/AuthProvider'
import { Router } from './routes/Router'

export const App: React.FC = () => {
    return (
        <AuthProvider>
            <div style={{ fontFamily: 'Inter, system-ui, Arial', lineHeight: 1.4 }}>
                <Router />
            </div>
        </AuthProvider>
    )
}
