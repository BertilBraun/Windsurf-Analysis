import React from 'react'

const PAYPAL_LINK = 'https://paypal.me/bertilbraun'

function cx(...parts: Array<string | undefined | null | false>) {
    return parts.filter(Boolean).join(' ')
}

export type SupportProjectSectionProps = {
    className?: string
}

export const SupportProjectSection: React.FC<SupportProjectSectionProps> = ({ className }) => {
    return (
        <section className={cx('rounded-2xl border border-slate-200 bg-white p-6 sm:p-8', className)}>
            <h2 className="mt-0">Want to support the project?</h2>
            <div className="text-sm text-slate-600 leading-6">
                If GybeLock helps you, you can support its development. Any help is greatly appreciated!
            </div>
            <div className="mt-4 flex flex-wrap gap-3 items-center">
                <a
                    href={PAYPAL_LINK}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center justify-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition bg-slate-900 text-white hover:bg-slate-800"
                    title="Buy me a coffee (PayPal)"
                >
                    Buy me a coffee (PayPal)
                </a>
            </div>
        </section>
    )
}
