import React, { createContext, useCallback, useEffect, useMemo, useState } from 'react'
import {
    createUserWithEmailAndPassword,
    onIdTokenChanged,
    reload,
    sendEmailVerification,
    sendPasswordResetEmail,
    signInWithEmailAndPassword,
    signInWithPopup,
    signOut,
    type User,
} from 'firebase/auth'
import { auth, backendUrl, googleProvider } from '../../firebase'
import { useSettings } from '../hooks/useSettings'

type AuthContextValue = {
    isAuthenticated: boolean
    isSignedIn: boolean
    needsEmailVerification: boolean
    authHeader: string | null
    uid: string | null
    email: string | null
    login: (email: string, password: string) => Promise<void>
    signup: (email: string, password: string, password2: string) => Promise<void>
    loginWithGoogle: () => Promise<void>
    resetPassword: (email: string) => Promise<void>
    logout: () => void
    resendVerificationEmail: () => Promise<void>
    refreshVerificationStatus: () => Promise<void>
    getAuthHeader: (forceRefreshToken?: boolean) => Promise<string>
    authorizedFetch: (input: RequestInfo, init?: RequestInit) => Promise<Response>
    settings: ReturnType<typeof useSettings>['settings']
}

export const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export const API_BASE = backendUrl.replace(/\/+$/, '')

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { settings, clearAuth } = useSettings()
    const [user, setUser] = useState<User | null>(auth.currentUser)
    const [authHeader, setAuthHeader] = useState<string | null>(null)
    const [email, setEmail] = useState<string | null>(auth.currentUser?.email ?? null)

    const getAuthHeader = useCallback(async (forceRefreshToken?: boolean) => {
        const u = auth.currentUser
        if (!u) throw new Error('Not authenticated')
        const token = await u.getIdToken(!!forceRefreshToken)
        return `Bearer ${token}`
    }, [])

    const ensureBackendUser = useCallback(
        async (u?: User | null) => {
            const current = u ?? auth.currentUser
            if (!current) throw new Error('Not authenticated')
            const header = await getAuthHeader()
            const res = await fetch(`${API_BASE}/users/${current.uid}`, {
                method: 'POST',
                headers: { Authorization: header },
            })
            if (res.ok) return
            // Backend returns 400 if user already exists; treat as success (idempotent).
            if (res.status === 400) return
            throw new Error(await res.text())
        },
        [getAuthHeader]
    )

    const login = useCallback(async (e: string, p: string) => {
        await signInWithEmailAndPassword(auth, e.trim(), p)
        // Token + email are picked up via onIdTokenChanged below.
    }, [])

    const signup = useCallback(
        async (e: string, p: string, p2: string) => {
            const email = e.trim()
            if (!email) throw new Error('Email is required.')
            if (!p) throw new Error('Password is required.')
            if (p !== p2) throw new Error('Passwords do not match.')

            const result = await createUserWithEmailAndPassword(auth, email, p)
            // Send verification email for email/password signup
            await sendEmailVerification(result.user)
            // Always create the backend user record after signup
            await ensureBackendUser(result.user)
        },
        [ensureBackendUser]
    )

    const loginWithGoogle = useCallback(async () => {
        const result = await signInWithPopup(auth, googleProvider)
        // Always create the backend user record after (first-time) Google signup.
        // Calling this every time is fine; backend treats "already exists" as 400.
        await ensureBackendUser(result.user)
        // Token + email are picked up via onIdTokenChanged below.
    }, [ensureBackendUser])

    const resetPassword = useCallback(async (e: string) => {
        const targetEmail = e.trim()
        if (!targetEmail) throw new Error('Enter your email first.')
        await sendPasswordResetEmail(auth, targetEmail)
    }, [])

    const logout = useCallback(() => {
        void (async () => {
            await signOut(auth)
            await clearAuth() // remove any legacy stored basic auth (if present)
        })()
    }, [clearAuth])

    const resendVerificationEmailCb = useCallback(async () => {
        const u = auth.currentUser
        if (!u) throw new Error('Not signed in.')
        await sendEmailVerification(u)
    }, [])

    const refreshVerificationStatus = useCallback(async () => {
        const u = auth.currentUser
        if (!u) throw new Error('Not signed in.')
        await reload(u)
        // Force-refresh token so backend sees updated email_verified claim.
        await u.getIdToken(true)
        setUser(auth.currentUser)
    }, [])

    useEffect(() => {
        // Migration: wipe any previously stored Basic auth credentials from IndexedDB.
        void clearAuth()

        const unsub = onIdTokenChanged(auth, async u => {
            setUser(u)
            setEmail(u?.email ?? null)
            if (!u) {
                setAuthHeader(null)
                return
            }
            const token = await u.getIdToken()
            setAuthHeader(`Bearer ${token}`)
        })

        return () => unsub()
    }, [clearAuth])

    const authorizedFetch = useCallback(
        async (input: RequestInfo, init?: RequestInit) => {
            const header = await getAuthHeader()
            // Only allow relative API paths; prefix with apiBase
            const path = typeof input === 'string' ? input : (input as Request).url
            const url = `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`
            const res = await fetch(url, {
                ...init,
                headers: {
                    ...(init?.headers || {}),
                    Authorization: header,
                },
            })

            // if 401 go to login page
            if (res.status === 401) {
                await signOut(auth)
                await clearAuth()
                // Login is handled inside the Analyzer route.
                window.location.href = '/analyzer'
                throw new Error('Not authenticated')
            }
            return res
        },
        [clearAuth, getAuthHeader]
    )

    const value = useMemo<AuthContextValue>(
        () => ({
            isAuthenticated: !!user && !!user.emailVerified,
            isSignedIn: !!user,
            needsEmailVerification: !!user && !user.emailVerified,
            authHeader,
            uid: user?.uid ?? null,
            email,
            login,
            signup,
            loginWithGoogle,
            resetPassword,
            logout,
            resendVerificationEmail: resendVerificationEmailCb,
            refreshVerificationStatus,
            getAuthHeader,
            authorizedFetch,
            settings,
        }),
        [
            user,
            authHeader,
            email,
            login,
            signup,
            loginWithGoogle,
            resetPassword,
            logout,
            resendVerificationEmailCb,
            refreshVerificationStatus,
            getAuthHeader,
            authorizedFetch,
            settings,
        ]
    )

    return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
    const ctx = React.useContext(AuthContext)
    if (!ctx) throw new Error('useAuth must be used within AuthProvider')
    return ctx
}
