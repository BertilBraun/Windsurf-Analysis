import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/Button'

const PAYPAL_LINK = 'https://paypal.me/bertilbraun'

export const PricingPage: React.FC = () => {
    const navigate = useNavigate()
    return (
        <div className="max-w-3xl flex flex-col gap-6">
            <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
                <h1 className="mt-0">Pricing</h1>
                <div className="mt-1 text-sm text-slate-600">
                    GybeLock is currently in an early testing phase — simple and free while we improve quality.
                </div>
            </div>

            <section className="rounded-2xl border border-brand-600/20 bg-brand-50 p-6 sm:p-8">
                <div className="text-xs font-semibold text-brand-700">Beta phase</div>
                <div className="mt-1 text-lg font-semibold text-slate-900">Free to use</div>
                <div className="mt-3 text-sm text-slate-700 leading-6">
                    GybeLock is currently in an early testing phase.
                    <div className="mt-2">During this time:</div>
                    <ul className="mt-2 list-disc pl-5 space-y-1">
                        <li>All features are available</li>
                        <li>No payment is required</li>
                        <li>No credit card needed</li>
                    </ul>
                    <div className="mt-3">
                        We’re focusing on improving tracking quality, usability, and performance — and your feedback
                        helps shape what comes next.
                    </div>
                </div>
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
                <h2 className="mt-0">What happens after the beta?</h2>
                <div className="text-sm text-slate-600 leading-6">
                    After the testing phase (planned for <strong>March</strong>), GybeLock will move to a simple,
                    usage-based pricing model:
                    <ul className="mt-3 list-disc pl-5 space-y-1">
                        <li>Pay per processed video</li>
                        <li>No subscriptions</li>
                        <li>No long-term commitments</li>
                    </ul>
                    <div className="mt-3">
                        Before introducing pricing, we’ll ask our users for feedback to make sure it feels fair and
                        accessible.
                    </div>
                </div>
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
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

            <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8 flex flex-col sm:flex-row gap-4 sm:items-center">
                <div className="flex-1">
                    <div className="mt-1 text-lg font-semibold text-slate-900">Get started for free</div>
                    <div className="mt-1 text-sm text-slate-700">
                        <em>(5 free videos will remain even after the beta)</em>
                    </div>
                </div>
                <Button variant="primary" onClick={() => navigate('/analyzer')} text="Get started free" />
            </section>
        </div>
    )
}
