import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import en from './locales/en.json'
import de from './locales/de.json'
import es from './locales/es.json'
import it from './locales/it.json'

const SUPPORTED_LANGUAGES = ['en', 'de', 'es', 'it'] as const
type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number]
const DEFAULT_LANGUAGE: SupportedLanguage = 'en'
const STORAGE_KEY = 'gybelock.language'

const resources = {
    en: { translation: en },
    de: { translation: de },
    es: { translation: es },
    it: { translation: it },
} as const

function normalizeLanguage(lang: string | null | undefined): SupportedLanguage | null {
    if (!lang) return null
    const normalized = lang.toLowerCase().replace('_', '-').split('-')[0]
    return (SUPPORTED_LANGUAGES as readonly string[]).includes(normalized)
        ? (normalized as SupportedLanguage)
        : null
}

function getInitialLanguage(): SupportedLanguage {
    if (typeof window === 'undefined') return DEFAULT_LANGUAGE
    const stored = normalizeLanguage(window.localStorage.getItem(STORAGE_KEY))
    if (stored) return stored
    return normalizeLanguage(window.navigator?.language) ?? DEFAULT_LANGUAGE
}

i18n.use(initReactI18next).init({
    resources,
    fallbackLng: DEFAULT_LANGUAGE,
    supportedLngs: SUPPORTED_LANGUAGES,
    lng: getInitialLanguage(),
    load: 'languageOnly',
    initImmediate: false,
    interpolation: { escapeValue: false },
    react: { useSuspense: false, bindI18n: 'languageChanged' },
})

i18n.on('languageChanged', language => {
    if (typeof window === 'undefined') return
    const normalized = normalizeLanguage(language) ?? DEFAULT_LANGUAGE
    window.localStorage.setItem(STORAGE_KEY, normalized)
})

export default i18n
