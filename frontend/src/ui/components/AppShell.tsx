import React from 'react'
import { Link, NavLink, Outlet } from 'react-router-dom'
import { LogoButton } from './LogoButton'

export const AppShellLayout: React.FC = () => {
    return (
        <div className="min-h-dvh bg-white text-slate-900">
            <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                    <LogoButton to="/" imgStyle={{ imageRendering: 'auto' }} />

                    <div className="flex-1" />

                    <nav className="flex items-center gap-1" aria-label="Primary navigation">
                        <TopNav to="/" label="Home" />
                        <TopNav to="/analyzer" label="Analyzer" />
                        <TopNav to="/pricing" label="Pricing" />
                        <TopNav to="/faq" label="FAQ" />
                    </nav>
                </div>
            </header>

            <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-6">
                <Outlet />
            </main>

            <footer className="border-t border-slate-200 max-w-[1400px] margin-auto">
                <div className="text-sm text-slate-500 mx-auto text-center py-6">
                    Special thanks to Felix for all the Support and Ideas.
                </div>
                <div className="mx-auto px-4 sm:px-6 pb-6 flex flex-col sm:flex-row gap-3 sm:items-center mr-10">
                    <div className="text-xs text-slate-500">© {new Date().getFullYear()} GybeLock</div>
                    <div className="flex-1" />
                    <div className="flex flex-wrap gap-x-4 gap-y-2">
                        <FooterLink to="/terms" label="Terms of Use" />
                        <FooterLink to="/privacy" label="Privacy Policy" />
                        <FooterLink to="/impressum" label="Impressum" />
                        <FooterLink to="/contact" label="Contact" />
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

const FooterLink: React.FC<{ to: string; label: string }> = ({ to, label }) => {
    return (
        <Link to={to} className="text-xs text-slate-600 hover:text-slate-900 underline-offset-4 hover:underline">
            {label}
        </Link>
    )
}
