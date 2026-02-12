/**
 * @file GetStartedSection.tsx
 * @description Provides a call-to-action section for the landing page, directing users to key application features.
 */

import React from 'react'
import { Button } from './Button'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { Heading, Text } from './Typography'

/**
 * A promotional section component that encourages users to start using the application.
 *
 * This component displays a headline and description along with primary and secondary
 * action buttons that navigate to the demo and analyzer routes respectively.
 *
 * @returns {JSX.Element} The rendered call-to-action section.
 */
export const GetStartedSection: React.FC = () => {
    const { t } = useTranslation()
    const navigate = useNavigate()

    return (
        <section className="rounded-2xl border border-brand-600/20 bg-brand-50 p-6 sm:p-8 flex flex-col sm:flex-row gap-4 sm:items-center">
            <div className="flex-1">
                <Heading level={3} tone="brand">
                    {t('screens.home.ctaSection.headline')}
                </Heading>
                <Text as="div" variant="support">
                    {t('screens.home.ctaSection.body')}
                </Text>
            </div>
            <div className="flex flex-col sm:flex-row gap-2">
                <Button variant="primary" onClick={() => navigate('/demo')} text={t('screens.home.ctaSection.cta')} />
                <Button
                    variant="outline"
                    onClick={() => navigate('/analyzer')}
                    text={t('screens.home.ctaSection.ctaSecondary')}
                />
                <Text as="div" variant="muted" className="sm:hidden">
                    {t('common.fullAnalyzerBetaFreeNote')}
                </Text>
            </div>
            {/*<Text as="div" variant="muted" className="hidden sm:block sm:w-[220px] text-right">
                {t('common.fullAnalyzerBetaFreeNote')}
            </Text>*/}
        </section>
    )
}
