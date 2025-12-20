import React from 'react'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../auth/AuthProvider'

export const SignupPage: React.FC<{ onBackToLogin: () => void; onSuccess: () => void }> = ({
    onBackToLogin,
    onSuccess,
}) => {
    const { t } = useTranslation()
    const { signup } = useAuth()
    const [email, setEmail] = React.useState('')
    const [password, setPassword] = React.useState('')
    const [password2, setPassword2] = React.useState('')
    const [error, setError] = React.useState<string | null>(null)
    const [info, setInfo] = React.useState<string | null>(null)
    const [isSubmitting, setIsSubmitting] = React.useState(false)

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError(null)
        setInfo(null)
        setIsSubmitting(true)
        try {
            await signup(email, password, password2)
            setInfo(t('screens.signup.info.created'))
            onSuccess()
        } catch (err: any) {
            setError(String(err?.message || t('screens.signup.errors.failed')))
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <div style={{ maxWidth: 480 }}>
            <h3>{t('screens.signup.title')}</h3>
            <form onSubmit={handleSubmit}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <input
                        placeholder={t('screens.signup.placeholders.email')}
                        value={email}
                        onChange={e => setEmail(e.target.value)}
                        autoComplete="email"
                    />
                    <input
                        placeholder={t('screens.signup.placeholders.password')}
                        type="password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        autoComplete="new-password"
                    />
                    <input
                        placeholder={t('screens.signup.placeholders.repeatPassword')}
                        type="password"
                        value={password2}
                        onChange={e => setPassword2(e.target.value)}
                        autoComplete="new-password"
                    />
                    {info && <div style={{ color: '#0f766e', fontSize: 12 }}>{info}</div>}
                    {error && <div style={{ color: '#ef4444' }}>{error}</div>}
                    <button type="submit" disabled={!email || !password || !password2 || isSubmitting}>
                        {t('screens.signup.actions.createAccount')}
                    </button>
                </div>
            </form>
            <div style={{ marginTop: 12 }}>
                <button onClick={onBackToLogin} style={{ fontSize: 12 }}>
                    {t('screens.signup.actions.backToLogin')}
                </button>
            </div>
        </div>
    )
}
