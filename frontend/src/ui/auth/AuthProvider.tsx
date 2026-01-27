/**
 * @file AuthProvider.tsx
 * @description Provides authentication context and utilities using Firebase Auth,
 * managing user sessions, registration, and authorized API communication.
 */

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

/**
 * Shape of the authentication context value, providing user state and auth methods.
 */
type AuthContextValue = {
    /** The current Firebase user object. */
    user: User | null
    /** Whether the initial authentication state has been loaded from Firebase. */
    isAuthReady: boolean
    /** Whether the user is signed in and has a verified email. */
    isAuthenticated: boolean
    /** Whether the user is signed in (regardless of email verification). */
    isSignedIn: boolean
    /** Whether the user is signed in but needs to verify their email. */
    needsEmailVerification: boolean
    /** The current Bearer token header string. */
    authHeader: string | null
    /** The unique identifier for the user. */
    uid: string | null
    /** The user's email address. */
    email: string | null
    /** Logs in a user with email and password. */
    login: (email: string, password: string) => Promise<void>
    /** Signs up a new user, sends verification email, and creates a backend record. */
    signup: (
        email: string,
        password: string,
        password2: string,
        consent?: { termsAccepted: boolean; marketingConsent: boolean }
    ) => Promise<void>
    /** Logs in or signs up a user using Google OAuth. */
    loginWithGoogle: () => Promise<void>
    /** Sends a password reset email to the specified address. */
    resetPassword: (email: string) => Promise<void>
    /** Signs out the current user and clears local auth state. */
    logout: () => void
    /** Resends the email verification link to the current user. */
    resendVerificationEmail: () => Promise<void>
    /** Reloads the user profile to check for updated verification status. */
    refreshVerificationStatus: () => Promise<void>
    /** Performs a fetch request with the current authentication token injected into headers. */
    authorizedFetch: (input: RequestInfo, init?: RequestInit) => Promise<Response>
    /** Current application settings. */
    settings: ReturnType<typeof useSettings>['settings']
}

/**
 * Context for accessing authentication state and methods.
 */
export const AuthContext = createContext<AuthContextValue | undefined>(undefined)

/**
 * The base URL for the backend API, derived from the environment configuration.
 */
export const API_BASE = backendUrl.replace(/\/+$/, '')

/**
 * Provider component that manages authentication state using Firebase.
 * Handles login, signup, password reset, and authorized API requests.
 *
 * @param props - Component props.
 * @param props.children - Child components that will have access to the auth context.
 */
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
                // Login is handled inside the Analyzer/Demo routes.
                window.location.href = window.location.pathname.startsWith('/demo') ? '/demo' : '/analyzer'
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

/**
 * Hook to access the authentication context.
 *
 * @returns The current authentication context value.
 * @throws Error if used outside of an AuthProvider.
 */
export function useAuth(): AuthContextValue {
    const ctx = React.useContext(AuthContext)
    if (!ctx) throw new Error('useAuth must be used within AuthProvider')
    return ctx
}
