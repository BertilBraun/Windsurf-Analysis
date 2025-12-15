import React, { createContext, useCallback, useEffect, useMemo, useState } from 'react'
import { useSettings } from '../hooks/useSettings'

type AuthContextValue = {
    isAuthenticated: boolean
    authHeader: string | null
    email: string | null
    login: (email: string, password: string) => Promise<void>
    logout: () => void
    authorizedFetch: (input: RequestInfo, init?: RequestInit) => Promise<Response>
    settings: ReturnType<typeof useSettings>['settings']
}

export const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export const API_BASE = 'https://bertil-braun-private--windsurf-analysis-fastapi-app.modal.run/api/v1'

function makeAuthHeader(email: string, password: string): string {
    return 'Basic ' + btoa(`${email}:${password}`)
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { settings, setAuth, clearAuth } = useSettings()

    const login = useCallback(async (e: string, p: string) => {
        const h = makeAuthHeader(e, p)
        // Try verifying credentials before committing them
        const res = await fetch(`${API_BASE}/admin/verify`, {
            method: 'GET',
            headers: { Authorization: h },
        })
        if (!res.ok) {
            throw new Error('Invalid credentials')
        }
        await setAuth(e, h)
    }, [])

    const logout = useCallback(() => {
        clearAuth()
    }, [])

    const authorizedFetch = useCallback(
        async (input: RequestInfo, init?: RequestInit) => {
            if (!settings.authHeader) throw new Error('Not authenticated')
            // Only allow relative API paths; prefix with apiBase
            const path = typeof input === 'string' ? input : (input as Request).url
            const url = `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`
            const res = await fetch(url, {
                ...init,
                headers: {
                    ...(init?.headers || {}),
                    Authorization: settings.authHeader,
                },
            })

            // if 401 go to login page
            if (res.status === 401) {
                clearAuth()
                // Login is handled inside the Analyzer route.
                window.location.href = '/analyzer'
                throw new Error('Not authenticated')
            }
            return res
        },
        [settings.authHeader]
    )

    const value = useMemo<AuthContextValue>(
        () => ({
            isAuthenticated: !!settings.authHeader,
            authHeader: settings.authHeader,
            email: settings.authEmail,
            login,
            logout,
            authorizedFetch,
            settings,
        }),
        [settings.authHeader, settings.authEmail, login, logout, authorizedFetch]
    )

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
    const ctx = React.useContext(AuthContext)
    if (!ctx) throw new Error('useAuth must be used within AuthProvider')
    return ctx
}
