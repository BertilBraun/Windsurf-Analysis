import React from 'react'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../auth/AuthProvider'

export const LoginPage: React.FC<{ onSignup: () => void; onSuccess: () => void }> = ({ onSignup, onSuccess }) => {
    const { t } = useTranslation()
    const { login, loginWithGoogle, resetPassword } = useAuth()
    const [email, setEmail] = React.useState('')
    const [password, setPassword] = React.useState('')
    const [error, setError] = React.useState<string | null>(null)
    const [info, setInfo] = React.useState<string | null>(null)
    const [isSubmitting, setIsSubmitting] = React.useState(false)

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
                    />
                    <input
                        placeholder={t('screens.login.placeholders.password')}
                        type="password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                    />
                    {info && <div style={{ color: '#0f766e', fontSize: 12 }}>{info}</div>}
                    {error && <div style={{ color: '#ef4444' }}>{error}</div>}
                    <button type="submit" disabled={!email || !password || isSubmitting}>
                        {t('screens.login.actions.login')}
                    </button>
                    <button
                        type="button"
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
                    >
                        {t('screens.login.actions.google')}
                    </button>
                    <button
                        type="button"
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
                    >
                        {t('screens.login.actions.forgot')}
                    </button>
                </div>
            </form>
            <div style={{ marginTop: 12 }}>
                <button onClick={onSignup} style={{ fontSize: 12 }}>
                    {t('screens.login.actions.createAccount')}
                </button>
            </div>
        </div>
    )
}
