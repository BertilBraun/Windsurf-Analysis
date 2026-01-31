/**
 * @file UnsupportedBrowserPage.tsx
 * @module UnsupportedBrowserPage
 * @description Dedicated screen shown when the browser is missing required APIs.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from '../components/Button'
import { LogoButton } from '../components/LogoButton'

export const UnsupportedBrowserPage: React.FC<{
    onGoHome: () => void
    onGoDemo: () => void
}> = ({ onGoHome, onGoDemo }) => {
    const { t } = useTranslation()

    return (
        <div className="min-h-dvh bg-white text-slate-900">
            <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur">
                <div className="mx-auto max-w-[1400px] px-4 sm:px-6 py-3 flex items-center gap-3">
                    <LogoButton onClick={onGoHome} />
                    <div className="flex-1" />
                </div>
            </header>

            <main className="mx-auto max-w-[1400px] px-4 sm:px-6 py-10">
                <div className="max-w-2xl rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-3">
                    <h1 className="m-0 text-xl font-semibold tracking-tight">
                        {t('components.unsupportedBrowser.title')}
                    </h1>

                    <p className="m-0 text-sm text-slate-700">{t('components.unsupportedBrowser.body')}</p>
                    <p className="m-0 text-sm text-slate-700">
                        {t('components.unsupportedBrowser.recommendation')}
                    </p>

                    <div className="flex flex-wrap gap-2 pt-2">
                        <Button
                            type="button"
                            variant="primary"
                            size="md"
                            onClick={onGoDemo}
                            text={t('components.unsupportedBrowser.actions.tryDemo')}
                        />
                        <Button
                            type="button"
                            variant="outline"
                            size="md"
                            onClick={onGoHome}
                            text={t('components.unsupportedBrowser.actions.backHome')}
                        />
                    </div>
                </div>
            </main>
        </div>
    )
}

