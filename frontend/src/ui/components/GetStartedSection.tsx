import React from 'react'
import { Button } from './Button'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'

export const GetStartedSection: React.FC = () => {
    const { t } = useTranslation()
    const navigate = useNavigate()

    return (
        <section className="rounded-2xl border border-brand-600/20 bg-brand-50 p-6 sm:p-8 flex flex-col sm:flex-row gap-4 sm:items-center">
            <div className="flex-1">
                <div className="text-xs font-semibold text-brand-700">{t('screens.home.ctaSection.title')}</div>
                <div className="mt-1 text-lg font-semibold text-slate-900">{t('screens.home.ctaSection.headline')}</div>
                <div className="mt-1 text-sm text-slate-700">{t('screens.home.ctaSection.body')}</div>
            </div>
            <Button variant="primary" onClick={() => navigate('/analyzer')} text={t('screens.home.ctaSection.cta')} />
        </section>
    )
}
