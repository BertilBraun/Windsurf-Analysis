import React from 'react'
import { useTranslation } from 'react-i18next'
import flagDe from '../assets/flags/de.svg'
import flagEn from '../assets/flags/en.svg'
import flagEs from '../assets/flags/es.svg'
import flagIt from '../assets/flags/it.svg'

type LanguageOption = {
    code: 'en' | 'de' | 'es' | 'it'
    name: string
    flagSrc: string
}

const LANGUAGES: LanguageOption[] = [
    { code: 'en', name: 'English', flagSrc: flagEn },
    { code: 'de', name: 'Deutsch', flagSrc: flagDe },
    { code: 'es', name: 'Espanol', flagSrc: flagEs },
    { code: 'it', name: 'Italiano', flagSrc: flagIt },
]

function cx(...parts: Array<string | undefined | null | false>) {
    return parts.filter(Boolean).join(' ')
}

function normalizeLanguage(lang: string | undefined | null) {
    if (!lang) return 'en'
    const normalized = lang.toLowerCase().replace('_', '-')
    return normalized.split('-')[0]
}

export const LanguageSwitcher: React.FC<{ className?: string }> = ({ className }) => {
    const { i18n, t } = useTranslation()
    const [open, setOpen] = React.useState(false)
    const containerRef = React.useRef<HTMLDivElement | null>(null)
    const current = normalizeLanguage(i18n.language)
    const activeCode = LANGUAGES.some(language => language.code === current) ? current : 'en'
    const active = LANGUAGES.find(language => language.code === activeCode) ?? LANGUAGES[0]

    React.useEffect(() => {
        const onPointerDown = (event: MouseEvent) => {
            if (!containerRef.current) return
            if (containerRef.current.contains(event.target as Node)) return
            setOpen(false)
        }
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Escape') setOpen(false)
        }
        document.addEventListener('mousedown', onPointerDown)
        document.addEventListener('keydown', onKeyDown)
        return () => {
            document.removeEventListener('mousedown', onPointerDown)
            document.removeEventListener('keydown', onKeyDown)
        }
    }, [])

    return (
        <div ref={containerRef} className={cx('relative', className)}>
            <button
                type="button"
                aria-label={t('components.languageSwitcher.ariaLabel', { language: active.name })}
                aria-haspopup="menu"
                aria-expanded={open}
                onClick={() => setOpen(openState => !openState)}
                className={cx(
                    'inline-flex items-center gap-2 rounded-md border border-slate-200 bg-white/90 px-2 py-1 text-xs text-slate-700',
                    'hover:bg-white focus:outline-none focus:ring-2 focus:ring-brand-500/30'
                )}
            >
                <img
                    src={active.flagSrc}
                    alt=""
                    className="h-4 w-6 rounded-sm border border-slate-200"
                    loading="lazy"
                />
                <svg
                    width="10"
                    height="10"
                    viewBox="0 0 10 10"
                    aria-hidden="true"
                    className={cx('text-slate-500 transition', open ? 'rotate-180' : undefined)}
                >
                    <path d="M2 3.5L5 6.5L8 3.5" fill="none" stroke="currentColor" strokeWidth="1.3" />
                </svg>
            </button>

            {open && (
                <div
                    role="menu"
                    aria-label={t('components.languageSwitcher.menuLabel')}
                    className="absolute right-0 mt-2 w-36 rounded-md border border-slate-200 bg-white shadow-lg z-50"
                >
                    {LANGUAGES.map(language => {
                        const isActive = language.code === activeCode
                        return (
                            <button
                                key={language.code}
                                type="button"
                                role="menuitem"
                                onClick={() => {
                                    setOpen(false)
                                    void i18n.changeLanguage(language.code)
                                }}
                                className={cx(
                                    'w-full px-2 py-1.5 text-xs text-left flex items-center gap-2',
                                    isActive ? 'bg-slate-100 text-slate-900' : 'text-slate-700 hover:bg-slate-50'
                                )}
                            >
                                <img
                                    src={language.flagSrc}
                                    alt=""
                                    className="h-4 w-6 rounded-sm border border-slate-200"
                                    loading="lazy"
                                />
                                <span>{language.name}</span>
                            </button>
                        )
                    })}
                </div>
            )}
        </div>
    )
}
