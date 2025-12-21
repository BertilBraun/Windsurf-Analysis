import React from 'react'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../auth/AuthProvider'
import { Button } from '../components/Button'

export const LoginPage: React.FC<{ onSignup: () => void; onSuccess: () => void }> = ({ onSignup, onSuccess }) => {
    const { t } = useTranslation()
    const { login, loginWithGoogle, resetPassword } = useAuth()
    const [email, setEmail] = React.useState('')
    const [password, setPassword] = React.useState('')
    const [error, setError] = React.useState<string | null>(null)
    const [info, setInfo] = React.useState<string | null>(null)
    const [isSubmitting, setIsSubmitting] = React.useState(false)
    const inputClassName =
        'w-full rounded-md border border-slate-300 bg-white/90 px-3 py-2 text-sm shadow-sm focus:border-brand-600 focus:outline-none focus:ring-2 focus:ring-brand-600/20'

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
        <div style={{ maxWidth: 420 }}>
            <h3>{t('screens.login.title')}</h3>
            <form onSubmit={handleSubmit}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <input
                        placeholder={t('screens.login.placeholders.email')}
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        className={inputClassName}
                    />
                    <input
                        placeholder={t('screens.login.placeholders.password')}
                        type="password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        className={inputClassName}
                    />
                    {info && <div style={{ color: '#0f766e', fontSize: 12 }}>{info}</div>}
                    {error && <div style={{ color: '#ef4444' }}>{error}</div>}
                    <Button
                        type="submit"
                        text={t('screens.login.actions.login')}
                        disabled={!email || !password || isSubmitting}
                        isPending={isSubmitting}
                    />
                    <Button
                        type="button"
                        text={t('screens.login.actions.google')}
                        disabled={isSubmitting}
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
                    />
                    <Button
                        type="button"
                        text={t('screens.login.actions.forgot')}
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
                        style={{ fontSize: 12 }}
                    />
                </div>
            </form>
            <div style={{ marginTop: 12 }}>
                <Button
                    variant="unstyled"
                    size="none"
                    onClick={onSignup}
                    style={{ fontSize: 12 }}
                    text={t('screens.login.actions.createAccount')}
                />
            </div>
        </div>
    )
}
