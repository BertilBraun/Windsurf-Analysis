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
    const bannerRef = React.useRef<HTMLDivElement | null>(null)
    const previousOffsetRef = React.useRef<string | null>(null)

    React.useEffect(() => {
        // Read consent lazily to avoid SSR pitfalls (even though this is a SPA).
        setConsent(getAnalyticsConsent())
    }, [])

    React.useLayoutEffect(() => {
        const shouldShow = location.pathname !== '/privacy' && consent === null
        if (!shouldShow) {
            return
        }

        const banner = bannerRef.current
        if (!banner) return

        const applyOffset = () => {
            const height = Math.ceil(banner.getBoundingClientRect().height)
            document.documentElement.style.setProperty('--analytics-consent-offset', `${height}px`)
        }

        previousOffsetRef.current = document.documentElement.style.getPropertyValue('--analytics-consent-offset')
        applyOffset()

        let resizeObserver: ResizeObserver | null = null
        if (typeof ResizeObserver !== 'undefined') {
            resizeObserver = new ResizeObserver(() => applyOffset())
            resizeObserver.observe(banner)
        } else {
            window.addEventListener('resize', applyOffset)
        }

        return () => {
            if (resizeObserver) {
                resizeObserver.disconnect()
            } else {
                window.removeEventListener('resize', applyOffset)
            }
            if (previousOffsetRef.current) {
                document.documentElement.style.setProperty('--analytics-consent-offset', previousOffsetRef.current)
            } else {
                document.documentElement.style.removeProperty('--analytics-consent-offset')
            }
        }
    }, [consent, location.pathname])

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
        <div
            ref={bannerRef}
            className="fixed bottom-0 left-0 right-0 z-50 border-t border-slate-200 bg-white/95 backdrop-blur"
        >
            <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex flex-col sm:flex-row gap-3 sm:items-center">
                <div className="text-xs text-slate-700 leading-5">
                    <Trans
                        i18nKey="components.analyticsConsentBanner.message"
                        components={{
                            bold: <b />,
                            privacyLink: <Link className="text-brand-700 underline underline-offset-4" to="/privacy" />,
                        }}
                    />
                </div>
                <div className="flex-1" />
                <div className="flex gap-2">
                    <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => choose('declined')}
                        text={t('components.analyticsConsentBanner.decline')}
                    />
                    <Button
                        type="button"
                        variant="primary"
                        size="sm"
                        onClick={() => choose('accepted')}
                        text={t('components.analyticsConsentBanner.accept')}
                    />
                </div>
            </div>
        </div>
    )
}
