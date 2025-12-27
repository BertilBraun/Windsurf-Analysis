import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { SupportProjectSection } from '../components/SupportProjectSection'
import { GetStartedSection } from '../components/GetStartedSection'

export const PricingPage: React.FC = () => {
    const { t } = useTranslation()
    const navigate = useNavigate()
    return (
        <div className="max-w-3xl flex flex-col gap-6">
            <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
                <h1 className="mt-0">{t('screens.pricing.title')}</h1>
                <div className="mt-1 text-sm text-slate-600">{t('screens.pricing.intro')}</div>
                <div className="mt-3 text-sm text-slate-600 leading-6">
                    <Trans i18nKey="screens.pricing.lede" components={{ strong: <strong /> }} />
                </div>
            </div>

            <section className="rounded-2xl border border-brand-600/20 bg-brand-50 p-6 sm:p-8">
                <div className="text-xs font-semibold text-brand-700">{t('screens.pricing.beta.title')}</div>
                <div className="mt-1 text-lg font-semibold text-slate-900">{t('screens.pricing.beta.headline')}</div>
                <div className="mt-3 text-sm text-slate-700 leading-6">
                    {t('screens.pricing.beta.intro')}
                    <div className="mt-2">{t('screens.pricing.beta.during')}</div>
                    <ul className="mt-2 list-disc pl-5 space-y-1">
                        <li>{t('screens.pricing.beta.bullets.features')}</li>
                        <li>{t('screens.pricing.beta.bullets.payment')}</li>
                        <li>{t('screens.pricing.beta.bullets.card')}</li>
                    </ul>
                    <div className="mt-3">{t('screens.pricing.beta.outro')}</div>
                </div>
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
                <h2 className="mt-0">{t('screens.pricing.after.title')}</h2>
                <div className="text-sm text-slate-600 leading-6">
                    <Trans i18nKey="screens.pricing.after.intro" components={{ strong: <strong /> }} />
                    <ul className="mt-3 list-disc pl-5 space-y-1">
                        <li>{t('screens.pricing.after.bullets.perVideo')}</li>
                        <li>{t('screens.pricing.after.bullets.noSubscriptions')}</li>
                        <li>{t('screens.pricing.after.bullets.noCommitments')}</li>
                    </ul>
                    <div className="mt-3">{t('screens.pricing.after.outro')}</div>
                </div>
            </section>

            <SupportProjectSection />

            <GetStartedSection />
        </div>
    )
}
