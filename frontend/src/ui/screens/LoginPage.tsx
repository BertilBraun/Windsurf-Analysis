import React from 'react'
import { useAuth } from '../auth/AuthProvider'

export const LoginPage: React.FC<{ onSignup: () => void; onSuccess: () => void }> = ({ onSignup, onSuccess }) => {
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
            setError(String(err?.message || 'Login failed'))
        } finally {
            setIsSubmitting(false)
        }
    }

    return (
        <div style={{ maxWidth: 420 }}>
            <h3>Login</h3>
            <form onSubmit={handleSubmit}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <input placeholder="email" value={email} onChange={e => setEmail(e.target.value)} />
                    <input
                        placeholder="password"
                        type="password"
                        value={password}
                        onChange={e => setPassword(e.target.value)}
                    />
                    {info && <div style={{ color: '#0f766e', fontSize: 12 }}>{info}</div>}
                    {error && <div style={{ color: '#ef4444' }}>{error}</div>}
                    <button type="submit" disabled={!email || !password || isSubmitting}>
                        Login
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
                                setError(String(err?.message || 'Google sign-in failed'))
                            } finally {
                                setIsSubmitting(false)
                            }
                        }}
                    >
                        Sign in with Google
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
                                setInfo(`Password reset email sent to ${email.trim()}.`)
                            } catch (err: any) {
                                setError(String(err?.message || 'Could not send password reset email'))
                            } finally {
                                setIsSubmitting(false)
                            }
                        }}
                        style={{ fontSize: 12 }}
                    >
                        Forgot password
                    </button>
                </div>
            </form>
            <div style={{ marginTop: 12 }}>
                <button onClick={onSignup} style={{ fontSize: 12 }}>
                    Create an account
                </button>
            </div>
        </div>
    )
}
