import React from 'react'
import { useAuth } from '../auth/AuthProvider'

export const LoginPage: React.FC<{ onSignup: () => void; onSuccess: () => void }> = ({ onSignup, onSuccess }) => {
    const { login } = useAuth()
    const [email, setEmail] = React.useState('')
    const [password, setPassword] = React.useState('')
    const [error, setError] = React.useState<string | null>(null)
    const [isSubmitting, setIsSubmitting] = React.useState(false)

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError(null)
        setIsSubmitting(true)
        try {
            // Optimistic login; first request will fail if creds invalid
            login(email, password)
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
                    <input placeholder="password" type="password" value={password} onChange={e => setPassword(e.target.value)} />
                    {error && <div style={{ color: '#ef4444' }}>{error}</div>}
                    <button type="submit" disabled={!email || !password || isSubmitting}>Login</button>
                </div>
            </form>
            <div style={{ marginTop: 12 }}>
                <button onClick={onSignup} style={{ fontSize: 12 }}>Create an account</button>
            </div>
        </div>
    )
}


