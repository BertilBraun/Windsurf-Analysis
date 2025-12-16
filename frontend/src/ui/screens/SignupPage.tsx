import React from 'react'

export const SignupPage: React.FC<{ onBackToLogin: () => void }> = ({ onBackToLogin }) => {
    return (
        <div style={{ maxWidth: 480 }}>
            <h3>Sign up</h3>
            <p>
                For now, sign-ups are manual. Please email{' '}
                <a href="mailto:bertil.braun.private@gmail.com">bertil.braun.private@gmail.com</a> to request access.
            </p>
            <button onClick={onBackToLogin}>Back to login</button>
        </div>
    )
}
