import React from 'react'
import { AuthProvider } from './auth/AuthProvider'
import { Router } from './routes/Router'

export const App: React.FC = () => {
    return (
        <AuthProvider>
            <div style={{ fontFamily: 'Inter, system-ui, Arial', margin: '24px', lineHeight: 1.4 }}>
                <h2>Windsurf Analysis</h2>
                <Router />
            </div>
        </AuthProvider>
    )
}
