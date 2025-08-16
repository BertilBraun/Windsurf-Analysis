import React, { createContext, useCallback, useEffect, useMemo, useState } from 'react'

type AuthContextValue = {
    isAuthenticated: boolean
    authHeader: string | null
    email: string | null
    login: (email: string, password: string) => void
    logout: () => void
    authorizedFetch: (input: RequestInfo, init?: RequestInit) => Promise<Response>
}

export const AuthContext = createContext<AuthContextValue | undefined>(undefined)

const STORAGE_KEY = 'windsurf_auth'
export const API_BASE = 'https://bertil-braun-private--windsurf-analysis-fastapi-app.modal.run/api/v1'

function makeAuthHeader(email: string, password: string): string {
    return 'Basic ' + btoa(`${email}:${password}`)
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [authHeader, setAuthHeader] = useState<string | null>(null)
    const [email, setEmail] = useState<string | null>(null)

    useEffect(() => {
        try {
            const raw = localStorage.getItem(STORAGE_KEY)
            if (raw) {
                const { email: e, authHeader: h } = JSON.parse(raw) as { email: string; authHeader: string }
                setEmail(e)
                setAuthHeader(h)
            }
        } catch {}
    }, [])

    const login = useCallback((e: string, p: string) => {
        const h = makeAuthHeader(e, p)
        setAuthHeader(h)
        setEmail(e)
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify({ email: e, authHeader: h }))
        } catch {}
    }, [])

    const logout = useCallback(() => {
        setAuthHeader(null)
        setEmail(null)
        try {
            localStorage.removeItem(STORAGE_KEY)
        } catch {}
    }, [])

    const authorizedFetch = useCallback(
        async (input: RequestInfo, init?: RequestInit) => {
            if (!authHeader) throw new Error('Not authenticated')
            // Only allow relative API paths; prefix with apiBase
            const path = typeof input === 'string' ? input : (input as Request).url
            const url = `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`
            const res = await fetch(url, {
                ...init,
                headers: {
                    ...(init?.headers || {}),
                    Authorization: authHeader,
                },
            })
            if (!res.ok) throw new Error(await res.text())
            return res
        },
        [authHeader]
    )

    const value = useMemo<AuthContextValue>(
        () => ({
            isAuthenticated: !!authHeader,
            authHeader,
            email,
            login,
            logout,
            authorizedFetch,
        }),
        [authHeader, email, login, logout, authorizedFetch]
    )

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
    const ctx = React.useContext(AuthContext)
    if (!ctx) throw new Error('useAuth must be used within AuthProvider')
    return ctx
}
