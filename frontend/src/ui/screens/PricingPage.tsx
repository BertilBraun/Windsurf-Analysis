/**
 * @file PricingPage.tsx
 * @description Pricing information page detailing the beta phase and future pay-per-use model.
 */

import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { SupportProjectSection } from '../components/SupportProjectSection'
import { GetStartedSection } from '../components/GetStartedSection'
import { Heading, Text, TextStack, TextStrong } from '../components/Typography'

/**
 * Renders the pricing page, providing details on the current beta phase,
 * future pay-per-use billing, and project support options.
 */
export const PricingPage: React.FC = () => {
    const { t } = useTranslation()
    return (
        <div className="max-w-3xl flex flex-col gap-6">
            <div className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
                <Heading level={1}>{t('screens.pricing.title')}</Heading>
                <Text as="div" variant="muted" className="mt-1">
                    {t('screens.pricing.intro')}
                </Text>
                <Text as="div" className="mt-3">
                    <Trans i18nKey="screens.pricing.lede" components={{ strong: <TextStrong /> }} />
                </Text>
            </div>

            <section className="rounded-2xl border border-brand-600/20 bg-brand-50 p-6 sm:p-8">
                <Text as="div" variant="muted" weight="semibold" tone="brand" className="mb-1">
                    {t('screens.pricing.beta.title')}
                </Text>
                <Heading level={3}>{t('screens.pricing.beta.headline')}</Heading>
                <TextStack>
                    <p>{t('screens.pricing.beta.intro')}</p>
                    <p>{t('screens.pricing.beta.during')}</p>
                    <ul className="list-disc pl-5 space-y-1">
                        <li>{t('screens.pricing.beta.bullets.features')}</li>
                        <li>{t('screens.pricing.beta.bullets.payment')}</li>
                        <li>{t('screens.pricing.beta.bullets.card')}</li>
                    </ul>
                    <p>{t('screens.pricing.beta.outro')}</p>
                </TextStack>
            </section>

            <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
                <Heading level={2}>{t('screens.pricing.after.title')}</Heading>
                <TextStack className="mt-3">
                    <p>
                        <Trans i18nKey="screens.pricing.after.intro" components={{ strong: <TextStrong /> }} />
                    </p>
                    <ul className="list-disc pl-5 space-y-1">
                        <li>{t('screens.pricing.after.bullets.perVideo')}</li>
                        <li>{t('screens.pricing.after.bullets.noSubscriptions')}</li>
                        <li>{t('screens.pricing.after.bullets.noCommitments')}</li>
                    </ul>
                    <p>{t('screens.pricing.after.outro')}</p>
                </TextStack>
            </section>

            <SupportProjectSection />

            <GetStartedSection />
        </div>
    )
}
