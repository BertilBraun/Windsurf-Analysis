import React from 'react'

export const LegalPage: React.FC<{ kind: 'terms' | 'privacy' | 'contact' }> = ({ kind }) => {
    const title = kind === 'terms' ? 'Terms of Use' : kind === 'privacy' ? 'Privacy Policy' : 'Contact'

    return (
        <div className="max-w-2xl">
            <h1>{title}</h1>
            {kind === 'contact' ? (
                <div className="text-sm text-slate-600 leading-6">
                    For help, feedback, or access requests, email{' '}
                    <a className="text-brand-700 underline" href="mailto:contact@gybelock.de">
                        contact@gybelock.de
                    </a>
                    .
                </div>
            ) : (
                <div className="text-sm text-slate-600 leading-6">
                    This page is a placeholder. Add your {title.toLowerCase()} content here when you’re ready.
                </div>
            )}
        </div>
    )
}
