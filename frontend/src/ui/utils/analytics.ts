/**
 * Google Analytics integration and consent management utilities.
 * Handles script initialization, event tracking, and automatic click monitoring.
 */

import { requireEnv } from '../../env'

declare global {
    interface Window {
        dataLayer?: unknown[]
        gtag?: (...args: any[]) => void
    }
}

type GtagParams = Record<string, unknown>

const CONSENT_KEY = 'gybelock_analytics_consent' as const

/**
 * User consent status for analytics tracking.
 */
export type AnalyticsConsent = 'accepted' | 'declined'

let _measurementId: string | null = null
let _initialized = false
let _clickTrackingInstalled = false

/**
 * Retrieves the stored analytics consent status from local storage.
 *
 * @returns The stored consent status, or null if no preference has been set.
 */
export function getAnalyticsConsent(): AnalyticsConsent | null {
    if (typeof window === 'undefined') return null
    try {
        const v = window.localStorage.getItem(CONSENT_KEY)
        if (v === 'accepted' || v === 'declined') return v
        return null
    } catch {
        return null
    }
}

/**
 * Persists the analytics consent status to local storage.
 *
 * @param consent - The consent status to store.
 */
export function setAnalyticsConsent(consent: AnalyticsConsent) {
    if (typeof window === 'undefined') return
    try {
        window.localStorage.setItem(CONSENT_KEY, consent)
    } catch {
        // ignore (e.g. private mode)
    }
}

/**
 * Checks if the user has explicitly accepted analytics tracking.
 *
 * @returns True if analytics are enabled.
 */
export function isAnalyticsEnabled(): boolean {
    return getAnalyticsConsent() === 'accepted'
}

function getMeasurementId(): string {
    return requireEnv('VITE_GA_MEASUREMENT_ID')
}

function ensureGtagDefined() {
    if (typeof window === 'undefined') return
    window.dataLayer = window.dataLayer || []
    window.gtag =
        window.gtag ||
        function gtagShim() {
            // eslint-disable-next-line prefer-rest-params
            window.dataLayer!.push(arguments as any)
        }
}

function ensureInitialized() {
    // Safe to call repeatedly; also makes sure early events aren't dropped due to effect ordering.
    if (!_initialized) initAnalytics()
}

/**
 * Initializes Google Analytics by loading the gtag.js script and configuring the measurement ID.
 * Only executes if analytics are enabled and the environment is a browser.
 */
export function initAnalytics() {
    if (typeof window === 'undefined' || typeof document === 'undefined') return
    if (!isAnalyticsEnabled()) return
    _measurementId = getMeasurementId()
    if (_initialized) return
    _initialized = true

    ensureGtagDefined()

    const existing = Array.from(document.querySelectorAll<HTMLScriptElement>('script[src]')).find(
        s => s.src.includes('www.googletagmanager.com/gtag/js?id=') && s.src.includes(_measurementId!)
    )
    if (!existing) {
        const s = document.createElement('script')
        s.async = true
        s.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(_measurementId)}`
        document.head.appendChild(s)
    }

    window.gtag?.('js', new Date())
    // We'll send page views manually on router navigation.
    window.gtag?.('config', _measurementId!, { send_page_view: false })
}

/**
 * Sets the user ID for the current analytics session to enable cross-device tracking.
 *
 * @param userId - The unique identifier for the user, or null to clear.
 */
export function setUserId(userId: string | null) {
    ensureInitialized()
    if (!_measurementId || !window?.gtag) return
    const uid = userId ? String(userId) : undefined
    window.gtag('set', { user_id: uid })
    window.gtag('config', _measurementId, { user_id: uid })
}

/**
 * Tracks a page view event for the specified path.
 *
 * @param pagePath - The URL path of the page being viewed.
 */
export function trackPageView(pagePath: string) {
    ensureInitialized()
    if (!_measurementId || !window?.gtag) return
    window.gtag('event', 'page_view', {
        page_path: pagePath,
        page_location: typeof window !== 'undefined' ? window.location.href : undefined,
    })
}

/**
 * Tracks a custom event with optional parameters.
 *
 * @param eventName - The name of the event to track.
 * @param params - Optional key-value pairs for event metadata.
 */
export function trackEvent(eventName: string, params?: GtagParams) {
    ensureInitialized()
    if (!_measurementId || !window?.gtag) return
    window.gtag('event', eventName, params || {})
}

function defaultClickLabel(el: HTMLElement): string | null {
    const explicit = el.getAttribute('data-analytics-label')
    if (explicit && explicit.trim()) return explicit.trim().slice(0, 120)
    const aria = el.getAttribute('aria-label')
    if (aria && aria.trim()) return aria.trim().slice(0, 120)
    const title = el.getAttribute('title')
    if (title && title.trim()) return title.trim().slice(0, 120)
    const text = (el.textContent || '').replace(/\s+/g, ' ').trim()
    if (!text) return null
    return text.slice(0, 120)
}

/**
 * Installs a global click listener to automatically track interactions with buttons,
 * links, and elements with ARIA button roles.
 */
export function installClickTracking() {
    ensureInitialized()
    if (typeof document === 'undefined') return
    if (!isAnalyticsEnabled()) return
    if (_clickTrackingInstalled) return
    _clickTrackingInstalled = true

    document.addEventListener(
        'click',
        e => {
            const target = e.target as HTMLElement | null
            if (!target) return

            const el =
                target.closest<HTMLElement>('[data-analytics-ignore="true"]') ||
                target.closest<HTMLElement>('button, a[href], [role="button"]')
            if (!el) return
            if (el.getAttribute('data-analytics-ignore') === 'true') return

            // Avoid tracking disabled buttons.
            if (el instanceof HTMLButtonElement && el.disabled) return
            if (el.getAttribute('aria-disabled') === 'true') return

            const label = defaultClickLabel(el)
            const kind = el.tagName.toLowerCase()
            trackEvent('ui_click', {
                kind,
                label: label ?? undefined,
                page_path: window.location?.pathname ?? undefined,
            })
        },
        true
    )
}
