import React from 'react'
import { useTranslation } from 'react-i18next'
import { useAuth } from '../auth/AuthProvider'
import { Button } from '../components/Button'

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
    const inputClassName =
        'w-full rounded-md border border-slate-300 bg-white/90 px-3 py-2 text-sm shadow-sm focus:border-brand-600 focus:outline-none focus:ring-2 focus:ring-brand-600/20'

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
                        className={inputClassName}
                    />
                    <input
                        placeholder={t('screens.signup.placeholders.password')}
                        type="password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                        autoComplete="new-password"
                        className={inputClassName}
                    />
                    <input
                        placeholder={t('screens.signup.placeholders.repeatPassword')}
                        type="password"
                        value={password2}
                        onChange={e => setPassword2(e.target.value)}
                        autoComplete="new-password"
                        className={inputClassName}
                    />
                    {info && <div style={{ color: '#0f766e', fontSize: 12 }}>{info}</div>}
                    {error && <div style={{ color: '#ef4444' }}>{error}</div>}
                    <Button
                        type="submit"
                        text={t('screens.signup.actions.createAccount')}
                        disabled={!email || !password || !password2 || isSubmitting}
                        isPending={isSubmitting}
                    />
                </div>
            </form>
            <div style={{ marginTop: 12 }}>
                <Button
                    variant="unstyled"
                    size="none"
                    onClick={onBackToLogin}
                    style={{ fontSize: 12 }}
                    text={t('screens.signup.actions.backToLogin')}
                />
            </div>
        </div>
    )
}
