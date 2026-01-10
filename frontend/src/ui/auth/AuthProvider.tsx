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
import { setUserId, trackEvent } from '../utils/analytics'

type AuthContextValue = {
    user: User | null
    isAuthReady: boolean
    isAuthenticated: boolean
    isSignedIn: boolean
    needsEmailVerification: boolean
    authHeader: string | null
    uid: string | null
    email: string | null
    login: (email: string, password: string) => Promise<void>
    signup: (
        email: string,
        password: string,
        password2: string,
        consent?: { termsAccepted: boolean; marketingConsent: boolean }
    ) => Promise<void>
    loginWithGoogle: () => Promise<void>
    resetPassword: (email: string) => Promise<void>
    logout: () => void
    resendVerificationEmail: () => Promise<void>
    refreshVerificationStatus: () => Promise<void>
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
    const [isAuthReady, setIsAuthReady] = useState(false)

    const getAuthHeader = useCallback(async (forceRefreshToken?: boolean) => {
        const u = auth.currentUser
        if (!u) throw new Error('Not authenticated')
        const token = await u.getIdToken(!!forceRefreshToken)
        return `Bearer ${token}`
    }, [])

    const ensureBackendUser = useCallback(
        async (u?: User | null, consent?: { termsAccepted: boolean; marketingConsent: boolean }) => {
            const current = u ?? auth.currentUser
            if (!current) throw new Error('Not authenticated')
            const header = await getAuthHeader()
            const hasConsent =
                !!consent && (consent.termsAccepted !== undefined || consent.marketingConsent !== undefined)
            const body = hasConsent
                ? JSON.stringify({
                      terms_accepted: consent.termsAccepted,
                      marketing_consent: consent.marketingConsent,
                  })
                : undefined
            const res = await fetch(`${API_BASE}/users/${current.uid}`, {
                method: 'POST',
                headers: {
                    Authorization: header,
                    ...(body ? { 'Content-Type': 'application/json' } : {}),
                },
                body,
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
        trackEvent('auth_login', { method: 'password' })
    }, [])

    const signup = useCallback(
        async (e: string, p: string, p2: string, consent?: { termsAccepted: boolean; marketingConsent: boolean }) => {
            const email = e.trim()
            if (!email) throw new Error('Email is required.')
            if (!p) throw new Error('Password is required.')
            if (p !== p2) throw new Error('Passwords do not match.')

            const result = await createUserWithEmailAndPassword(auth, email, p)
            // Send verification email for email/password signup
            await sendEmailVerification(result.user)
            // Always create the backend user record after signup
            await ensureBackendUser(result.user, consent)
            trackEvent('auth_signup', { method: 'password' })
        },
        [ensureBackendUser]
    )

    const loginWithGoogle = useCallback(async () => {
        const result = await signInWithPopup(auth, googleProvider)
        // Always create the backend user record after (first-time) Google signup.
        // Calling this every time is fine; backend treats "already exists" as 400.
        await ensureBackendUser(result.user)
        // Token + email are picked up via onIdTokenChanged below.
        trackEvent('auth_login', { method: 'google' })
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
            trackEvent('auth_logout')
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
            setIsAuthReady(true) // first callback means Firebase finished restoring auth state
            if (!u) {
                setAuthHeader(null)
                setUserId(null)
                return
            }
            const token = await u.getIdToken()
            setAuthHeader(`Bearer ${token}`)
            setUserId(u.uid)
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
            user,
            isAuthReady,
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
            isAuthReady,
            login,
            signup,
            loginWithGoogle,
            resetPassword,
            logout,
            resendVerificationEmailCb,
            refreshVerificationStatus,
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
