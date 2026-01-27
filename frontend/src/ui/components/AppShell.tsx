/**
 * @module AppShell
 * Provides the primary layout structure for the application, including the global header,
 * navigation menu, and footer.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { Link, NavLink, Outlet } from 'react-router-dom'
import { LanguageSwitcher } from './LanguageSwitcher'
import { LogoButton } from './LogoButton'
import { PAYPAL_LINK } from './SupportProjectSection'

/**
 * The main layout component that wraps the application's pages.
 *
 * It includes a sticky header with navigation links, a language switcher,
 * a main content area that renders nested routes via `<Outlet />`, and a footer.
 */
export const AppShellLayout: React.FC = () => {
    const { t } = useTranslation()
    return (
        <div className="min-h-dvh bg-white text-slate-900">
            <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                    <LogoButton to="/" imgStyle={{ imageRendering: 'auto' }} />

                    <div className="flex-1" />

                    <div className="flex items-center gap-2">
                        <nav className="flex items-center gap-1" aria-label={t('components.appShell.nav.ariaLabel')}>
                            <TopNav to="/" label={t('components.appShell.nav.home')} />
                            <AppMenu />
                            <TopNav to="/pricing" label={t('components.appShell.nav.pricing')} />
                            <TopNav to="/faq" label={t('components.appShell.nav.faq')} />
                            <TopNav to="/technical" label={t('components.appShell.nav.technical')} />
                        </nav>
                        <LanguageSwitcher />
                    </div>
                </div>
            </header>

            <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-6">
                <Outlet />
            </main>

            <footer className="border-t border-slate-200 max-w-[1400px] mx-auto">
                <div className="text-sm text-slate-500 mx-auto text-center py-6">
                    {t('components.appShell.footer.specialThanks')}
                </div>
                <div className="mx-auto px-4 sm:px-6 pb-6 flex flex-col sm:flex-row gap-3 sm:items-center mr-10">
                    <div className="text-xs text-slate-500">
                        {t('components.appShell.footer.copyright', { year: new Date().getFullYear() })}
                    </div>
                    <div className="flex-1" />
                    <div className="flex flex-wrap gap-x-4 gap-y-2">
                        <FooterLink to={PAYPAL_LINK} label={t('components.supportProjectSection.linkText')} />
                        <FooterLink to="/terms" label={t('components.appShell.footer.terms')} />
                        <FooterLink to="/privacy" label={t('components.appShell.footer.privacy')} />
                        <FooterLink to="/impressum" label={t('components.appShell.footer.impressum')} />
                        <FooterLink to="/contact" label={t('components.appShell.footer.contact')} />
                    </div>
                </div>
            </footer>
        </div>
    )
}

const TopNav: React.FC<{ to: string; label: string }> = ({ to, label }) => {
    return (
        <NavLink
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
                `text-sm px-3 py-1.5 rounded-md transition ${
                    isActive ? 'bg-brand-50 text-brand-700' : 'text-slate-700 hover:bg-slate-100'
                }`
            }
        >
            {label}
        </NavLink>
    )
}

const AppMenu: React.FC = () => {
    const { t } = useTranslation()
    const [open, setOpen] = React.useState(false)
    const rootRef = React.useRef<HTMLDivElement | null>(null)

    React.useEffect(() => {
        if (!open) return
        const onPointerDown = (event: MouseEvent) => {
            const target = event.target as Node | null
            if (!target || rootRef.current?.contains(target)) return
            setOpen(false)
        }
        document.addEventListener('mousedown', onPointerDown)
        return () => document.removeEventListener('mousedown', onPointerDown)
    }, [open])

    return (
        <div ref={rootRef} className="relative">
            <button
                type="button"
                className={`border-0 outline-none text-sm px-3 py-1.5 rounded-md transition ${
                    open ? 'bg-brand-50 text-brand-700' : 'text-slate-700 hover:bg-slate-100'
                }`}
                onClick={() => setOpen(v => !v)}
                aria-haspopup="menu"
                aria-expanded={open}
            >
                {t('components.appShell.nav.app')} <span aria-hidden="true">▾</span>
            </button>
            {open && (
                <div className="absolute left-0 mt-2 w-64 rounded-xl border border-slate-200 bg-white shadow-lg overflow-hidden z-50">
                    <MenuLink
                        to="/demo"
                        title={t('components.appShell.nav.appItems.demo.title')}
                        subtitle={t('components.appShell.nav.appItems.demo.subtitle')}
                        onPick={() => setOpen(false)}
                    />
                    <div className="h-px bg-slate-100" />
                    <MenuLink
                        to="/analyzer"
                        title={t('components.appShell.nav.appItems.analyzer.title')}
                        subtitle={t('components.appShell.nav.appItems.analyzer.subtitle')}
                        onPick={() => setOpen(false)}
                    />
                </div>
            )}
        </div>
    )
}

const MenuLink: React.FC<{ to: string; title: string; subtitle: string; onPick: () => void }> = ({
    to,
    title,
    subtitle,
    onPick,
}) => {
    return (
        <Link
            to={to}
            onClick={onPick}
            className="block px-4 py-3 hover:bg-slate-50 transition"
            role="menuitem"
        >
            <div className="text-sm font-semibold text-slate-900">{title}</div>
            <div className="text-xs text-slate-600 mt-0.5">{subtitle}</div>
        </Link>
    )
}

const FooterLink: React.FC<{ to: string; label: string }> = ({ to, label }) => {
    return (
        <Link to={to} className="text-xs text-slate-600 hover:text-slate-900 underline-offset-4 hover:underline">
            {label}
        </Link>
    )
}
