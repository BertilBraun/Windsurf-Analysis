import React from 'react'
import { Button } from './Button'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { Heading, Text } from './Typography'

export const GetStartedSection: React.FC = () => {
    const { t } = useTranslation()
    const navigate = useNavigate()

    return (
        <section className="rounded-2xl border border-brand-600/20 bg-brand-50 p-6 sm:p-8 flex flex-col sm:flex-row gap-4 sm:items-center">
            <div className="flex-1">
                <Text as="div" variant="muted" weight="semibold" tone="brand" className="mb-1">
                    {t('screens.home.ctaSection.title')}
                </Text>
                <Text as="div" variant="muted" className="mb-2">
                    {t('screens.home.ctaSection.note')}
                </Text>
                <Heading level={3}>{t('screens.home.ctaSection.headline')}</Heading>
                <Text as="div" variant="support">
                    {t('screens.home.ctaSection.body')}
                </Text>
            </div>
            <Button variant="primary" onClick={() => navigate('/analyzer')} text={t('screens.home.ctaSection.cta')} />
        </section>
    )
}
