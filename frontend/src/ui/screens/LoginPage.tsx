/**
 * @file LoginPage.tsx
 * @module LoginPage
 * @description Authentication screen providing login, Google OAuth, and password recovery.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../auth/AuthProvider'
import { Button } from '../components/Button'

const GoogleMark: React.FC<{ className?: string }> = ({ className }) => (
    <svg viewBox="0 0 48 48" className={className} aria-hidden="true">
        <path
            fill="#FFC107"
            d="M43.611 20.083H42V20H24v8h11.303C33.653 32.657 29.213 36 24 36c-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.965 3.035l5.657-5.657C34.994 6.053 29.77 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917Z"
        />
        <path
            fill="#FF3D00"
            d="M6.306 14.691 12.88 19.51C14.657 15.108 18.963 12 24 12c3.059 0 5.842 1.154 7.965 3.035l5.657-5.657C34.994 6.053 29.77 4 24 4 16.318 4 9.656 8.337 6.306 14.691Z"
        />
        <path
            fill="#4CAF50"
            d="M24 44c5.109 0 9.941-1.964 13.536-5.164l-6.279-5.313C29.247 35.052 26.735 36 24 36c-5.192 0-9.619-3.317-11.284-7.946l-6.523 5.026C9.504 39.556 16.227 44 24 44Z"
        />
        <path
            fill="#1976D2"
            d="M43.611 20.083H42V20H24v8h11.303a12.05 12.05 0 0 1-4.046 5.523l.003-.002 6.279 5.313C36.305 39.99 44 34 44 24c0-1.341-.138-2.65-.389-3.917Z"
        />
    </svg>
)

/**
 * Component for user authentication, supporting email/password login, Google OAuth, and password resets.
 *
 * @param props - Component properties.
 * @param props.onSignup - Callback triggered when the user navigates to the signup screen.
 * @param props.onSuccess - Callback triggered after a successful login.
 */
export const LoginPage: React.FC<{
    onSignup: () => void
    onSuccess: () => void
}> = ({ onSignup, onSuccess }) => {
    const { t } = useTranslation()
    const { login, loginWithGoogle, resetPassword } = useAuth()
    const [email, setEmail] = React.useState('')
    const [password, setPassword] = React.useState('')
    const [error, setError] = React.useState<string | null>(null)
    const [info, setInfo] = React.useState<string | null>(null)
    const [isSubmitting, setIsSubmitting] = React.useState(false)
    const inputClassName =
        'w-full rounded-xl border border-slate-200 bg-white/90 px-3 py-2.5 text-sm shadow-sm focus:border-brand-600 focus:outline-none focus:ring-2 focus:ring-brand-600/20'

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError(null)
        setInfo(null)
        setIsSubmitting(true)
        try {
            await login(email, password)
            onSuccess()
        } catch (err: any) {
            setError(String(err?.message || t('screens.login.errors.loginFailed')))
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <section className="rounded-3xl border border-slate-200 bg-white/90 backdrop-blur p-6 sm:p-8 shadow-lg">
            <div className="flex flex-col items-center text-center">
                <img src="/icon.png" alt="GybeLock" className="h-12 w-12" />
                <h1 className="m-0 mt-4 text-2xl font-semibold tracking-tight text-slate-900 leading-tight">
                    {t('screens.login.title')}
                </h1>
                <p className="mt-2 text-sm text-slate-600">{t('screens.login.subtitle')}</p>
            </div>

            <div className="mt-6">
                <Button
                    type="button"
                    variant="outline"
                    size="md"
                    className="w-full rounded-full py-3"
                    disabled={isSubmitting}
                    isPending={isSubmitting}
                    text={t('screens.login.actions.google')}
                    onClick={async () => {
                        setError(null)
                        setInfo(null)
                        setIsSubmitting(true)
                        try {
                            await loginWithGoogle()
                            onSuccess()
                        } catch (err: any) {
                            setError(String(err?.message || t('screens.login.errors.googleFailed')))
                        } finally {
                            setIsSubmitting(false)
                        }
                    }}
                >
                    <GoogleMark className="h-4 w-4" />
                    <span>{t('screens.login.actions.google')}</span>
                </Button>
            </div>

            <div className="relative my-6">
                <div className="h-px w-full bg-slate-200" />
                <div className="absolute inset-0 -top-2 flex items-center justify-center">
                    <span className="bg-white px-3 text-xs font-medium text-slate-500">{t('common.or')}</span>
                </div>
            </div>

            <form onSubmit={handleSubmit} className="flex flex-col gap-3">
                <label className="flex flex-col gap-1">
                    <span className="text-xs font-semibold text-slate-700">{t('screens.login.labels.email')}</span>
                    <input
                        placeholder={t('screens.login.placeholders.email')}
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        type="email"
                        autoComplete="email"
                        inputMode="email"
                        className={inputClassName}
                    />
                </label>

                <label className="flex flex-col gap-1">
                    <span className="text-xs font-semibold text-slate-700">{t('screens.login.labels.password')}</span>
                    <input
                        placeholder={t('screens.login.placeholders.password')}
                        type="password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        autoComplete="current-password"
                        className={inputClassName}
                    />
                </label>

                <div className="flex justify-end">
                    <Button
                        type="button"
                        variant="unstyled"
                        size="none"
                        className="border-0 bg-transparent p-0 text-xs text-slate-600 hover:text-slate-900 hover:bg-transparent underline underline-offset-4"
                        disabled={!email || isSubmitting}
                        onClick={async () => {
                            setError(null)
                            setInfo(null)
                            setIsSubmitting(true)
                            try {
                                await resetPassword(email)
                                setInfo(t('screens.login.info.resetSent', { email: email.trim() }))
                            } catch (err: any) {
                                setError(String(err?.message || t('screens.login.errors.resetFailed')))
                            } finally {
                                setIsSubmitting(false)
                            }
                        }}
                        text={t('screens.login.actions.forgot')}
                    />
                </div>

                {info && (
                    <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800">
                        {info}
                    </div>
                )}
                {error && (
                    <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
                        {error}
                    </div>
                )}

                <Button
                    type="submit"
                    variant="primary"
                    size="md"
                    className="w-full rounded-full py-3"
                    text={t('screens.login.actions.login')}
                    disabled={!email || !password || isSubmitting}
                    isPending={isSubmitting}
                />
            </form>

            <div className="mt-6 text-center text-sm text-slate-600">
                {t('screens.login.footer.noAccount')}{' '}
                <Button
                    type="button"
                    variant="unstyled"
                    size="none"
                    className="border-0 bg-transparent p-0 text-brand-700 hover:text-brand-600 hover:bg-transparent font-semibold underline underline-offset-4"
                    onClick={onSignup}
                    text={t('screens.login.footer.signup')}
                />
            </div>
        </section>
    )
}
