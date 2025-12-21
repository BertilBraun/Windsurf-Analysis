import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Trans, useTranslation } from 'react-i18next'
import {
    getAnalyticsConsent,
    initAnalytics,
    installClickTracking,
    setAnalyticsConsent,
    type AnalyticsConsent,
} from '../utils/analytics'
import { Button } from './Button'

function cx(...parts: Array<string | undefined | null | false>) {
    return parts.filter(Boolean).join(' ')
}

export const AnalyticsConsentBanner: React.FC = () => {
    const [consent, setConsent] = React.useState<AnalyticsConsent | null>(null)
    const location = useLocation()
    const { t } = useTranslation()

    React.useEffect(() => {
        // Read consent lazily to avoid SSR pitfalls (even though this is a SPA).
        setConsent(getAnalyticsConsent())
    }, [])

    // Don’t show banner on privacy page to keep it clean; the privacy page includes the same control.
    if (location.pathname === '/privacy') return null
    if (consent !== null) return null

    const choose = (v: AnalyticsConsent) => {
        setAnalyticsConsent(v)
        setConsent(v)
        if (v === 'accepted') {
            initAnalytics()
            installClickTracking()
        }
    }

    return (
        <div className="fixed bottom-0 left-0 right-0 z-50 border-t border-slate-200 bg-white/95 backdrop-blur">
            <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex flex-col sm:flex-row gap-3 sm:items-center">
                <div className="text-xs text-slate-700 leading-5">
                    <Trans
                        i18nKey="components.analyticsConsentBanner.message"
                        components={{
                            bold: <b />,
                            privacyLink: (
                                <Link className="text-brand-700 underline underline-offset-4" to="/privacy" />
                            ),
                        }}
                    />
                </div>
                <div className="flex-1" />
                <div className="flex gap-2">
                    <Button
                        type="button"
                        variant="unstyled"
                        size="none"
                        onClick={() => choose('declined')}
                        className={cx(
                            'rounded-lg border border-slate-200 px-3 py-2 text-xs',
                            'text-slate-800 hover:bg-slate-50'
                        )}
                        text={t('components.analyticsConsentBanner.decline')}
                    />
                    <Button
                        type="button"
                        variant="unstyled"
                        size="none"
                        onClick={() => choose('accepted')}
                        className={cx('rounded-lg bg-slate-900 text-white px-3 py-2 text-xs hover:bg-slate-800')}
                        text={t('components.analyticsConsentBanner.accept')}
                    />
                </div>
            </div>
        </div>
    )
}
