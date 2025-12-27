import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/Button'
import { Heading, Text, TextStrong } from '../components/Typography'

const FaqItem: React.FC<{ q: string; a: React.ReactNode }> = ({ q, a }) => {
    return (
        <details className="rounded-xl border border-slate-200 bg-white p-4">
            <summary className="cursor-pointer">
                <Text as="span" weight="semibold">
                    {q}
                </Text>
            </summary>
            <Text as="div" className="mt-2">
                {a}
            </Text>
        </details>
    )
}

export const FaqPage: React.FC = () => {
    const { t } = useTranslation()
    const navigate = useNavigate()
    return (
        <div className="flex flex-col gap-6">
            <div className="flex items-start justify-between gap-4">
                <div>
                    <Heading level={1}>{t('screens.faq.title')}</Heading>
                    <Text as="p" variant="muted" className="mt-1">
                        {t('screens.faq.subtitle')}
                    </Text>
                </div>
                <Button variant="primary" onClick={() => navigate('/analyzer')} text={t('screens.faq.cta')} />
            </div>

            <div className="flex flex-col gap-3">
                <FaqItem
                    q={t('screens.faq.items.whatDoes.question')}
                    a={
                        <Trans
                            i18nKey="screens.faq.items.whatDoes.answer"
                            components={{ strong: <TextStrong /> }}
                        />
                    }
                />
                <FaqItem
                    q={t('screens.faq.items.supportedFootage.question')}
                    a={
                        <Trans
                            i18nKey="screens.faq.items.supportedFootage.answer"
                            components={{ strong: <TextStrong /> }}
                        />
                    }
                />
                <FaqItem
                    q={t('screens.faq.items.ingressFolder.question')}
                    a={<Trans i18nKey="screens.faq.items.ingressFolder.answer" />} 
                />
                <FaqItem
                    q={t('screens.faq.items.processedVideos.question')}
                    a={<Trans i18nKey="screens.faq.items.processedVideos.answer" />} 
                />
                <FaqItem
                    q={t('screens.faq.items.cannotOpen.question')}
                    a={
                        <Trans
                            i18nKey="screens.faq.items.cannotOpen.answer"
                            components={{ strong: <TextStrong /> }}
                        />
                    }
                />
                <FaqItem
                    q={t('screens.faq.items.fileNotFound.question')}
                    a={
                        <Trans
                            i18nKey="screens.faq.items.fileNotFound.answer"
                            components={{ strong: <TextStrong /> }}
                        />
                    }
                />
                <FaqItem
                    q={t('screens.faq.items.movedVideo.question')}
                    a={
                        <Trans
                            i18nKey="screens.faq.items.movedVideo.answer"
                            components={{ strong: <TextStrong /> }}
                        />
                    }
                />
                <FaqItem
                    q={t('screens.faq.items.videoTooLong.question')}
                    a={<Trans i18nKey="screens.faq.items.videoTooLong.answer" />} 
                />
                <FaqItem
                    q={t('screens.faq.items.uploadSkipped.question')}
                    a={<Trans i18nKey="screens.faq.items.uploadSkipped.answer" />} 
                />
                <FaqItem
                    q={t('screens.faq.items.uploadsPaused.question')}
                    a={
                        <Trans
                            i18nKey="screens.faq.items.uploadsPaused.answer"
                            components={{ strong: <TextStrong /> }}
                        />
                    }
                />
                <FaqItem
                    q={t('screens.faq.items.formats.question')}
                    a={<Trans i18nKey="screens.faq.items.formats.answer" />} 
                />
                <FaqItem
                    q={t('screens.faq.items.shortcuts.question')}
                    a={
                        <Trans
                            i18nKey="screens.faq.items.shortcuts.answer"
                            components={{ strong: <TextStrong /> }}
                        />
                    }
                />
                <FaqItem
                    q={t('screens.faq.items.deleteAccount.question')}
                    a={
                        <Trans
                            i18nKey="screens.faq.items.deleteAccount.answer"
                            components={{ strong: <TextStrong /> }}
                        />
                    }
                />
            </div>
        </div>
    )
}
