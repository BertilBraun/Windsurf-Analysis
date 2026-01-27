/**
 * @file HomePage.tsx
 * @description Main landing page component for the application.
 */

import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/Button'
import { SupportProjectSection } from '../components/SupportProjectSection'
import { GetStartedSection } from '../components/GetStartedSection'
import { Heading, Text, TextStack, TextStrong } from '../components/Typography'

/**
 * The main landing page of the application.
 *
 * Displays the hero section with a video demo, explains the core problem and solution,
 * outlines the workflow steps, and provides links to the demo and support sections.
 */
export const HomePage: React.FC = () => {
    const { t } = useTranslation()
    const navigate = useNavigate()
    const [howStepKey, setHowStepKey] = React.useState<'ingress' | 'drop' | 'process' | 'review'>('ingress')
    const demoVideoSrc = React.useMemo(() => {
        const videos = [
            // { mp4: '/Surfer1.mp4', av1: '/Surfer1.av1.mp4' },
            // { mp4: '/Surfer2.mp4', av1: '/Surfer2.av1.mp4' },
            { mp4: '/SurferA.encoded.mp4', av1: '/SurferA.av1.mp4' },
            { mp4: '/SurferB.encoded.mp4', av1: '/SurferB.av1.mp4' },
            { mp4: '/SurferC.encoded.mp4', av1: '/SurferC.av1.mp4' },
        ]
        return videos[Math.floor(Math.random() * videos.length)]
    }, [])
    return (
        <div className="flex flex-col gap-10">
            <section className="rounded-2xl border border-slate-200 bg-linear-to-b from-white to-slate-50 p-5 sm:p-8">
                <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,2.3fr)] gap-6 items-center">
                    <div>
                        {/*<div className="text-xs font-semibold text-brand-700 mb-2">{t('screens.home.hero.brand')}</div>*/}
                        <Heading level={1}>{t('screens.home.hero.title')}</Heading>
                        <br />
                        <TextStack className="mt-4">
                            <p>
                                <Trans i18nKey="screens.home.hero.lede1" components={{ strong: <TextStrong /> }} />
                            </p>
                            <p>
                                <Trans i18nKey="screens.home.hero.lede2" components={{ strong: <TextStrong /> }} />
                            </p>
                            <p>{t('screens.home.hero.lede3')}</p>
                        </TextStack>

                        <br />
                        <div className="mt-5 flex flex-col sm:flex-row gap-3 sm:items-center">
                            <Button
                                variant="primary"
                                size="md"
                                onClick={() => navigate('/demo')}
                                text={t('screens.home.hero.cta')}
                            />
                        </div>
                        <br />
                        <Text as="div" variant="muted" className="mt-2">
                            {t('screens.home.hero.disclaimer')}
                        </Text>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white overflow-hidden">
                        <div className="aspect-video bg-slate-900 relative">
                            <video
                                key={demoVideoSrc.mp4}
                                className="w-full h-full object-cover"
                                autoPlay
                                loop
                                muted
                                playsInline
                            >
                                <source src={demoVideoSrc.av1} type='video/mp4; codecs="av01.0.05M.08"' />
                                <source src={demoVideoSrc.mp4} type="video/mp4" />
                            </video>
                            <div className="absolute inset-0 pointer-events-none select-none text-white font-bold drop-shadow">
                                <div className="absolute top-3 left-3 text-sm sm:text-base">Raw</div>
                                <div className="absolute bottom-3 left-3 text-sm sm:text-base">Tracking</div>
                                <div className="absolute top-3 right-3 text-sm sm:text-base">Result</div>
                            </div>
                        </div>
                        <div className="px-4 py-3 border-b border-slate-200 grid grid-cols-2 text-sm font-semibold text-slate-700">
                            <div className="text-center">{t('screens.home.comparison.beforeLabel')}</div>
                            <div className="text-center">{t('screens.home.comparison.afterLabel')}</div>
                        </div>
                    </div>
                </div>
            </section>

            <Section
                title={t('screens.home.problem.title')}
                imageSrc="/Camera on Beach.png"
                imageAlt={t('screens.home.images.problemAlt')}
                imageSide="left"
            >
                <p>{t('screens.home.problem.intro')}</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>{t('screens.home.problem.bullets.zoom')}</li>
                    <li>{t('screens.home.problem.bullets.riders')}</li>
                    <li>{t('screens.home.problem.bullets.moments')}</li>
                </ul>
                <p>{t('screens.home.problem.outro')}</p>
            </Section>

            <Section
                title={t('screens.home.what.title')}
                imageSrc="/Surfer Jump.jpg"
                imageAlt={t('screens.home.images.whatAlt')}
                imageSide="right"
            >
                <p>{t('screens.home.what.intro')}</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <Trans i18nKey="screens.home.what.bullets.stabilizes" components={{ strong: <TextStrong /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.home.what.bullets.locksOnto" components={{ strong: <TextStrong /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.home.what.bullets.centered" components={{ strong: <TextStrong /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.home.what.bullets.follows" components={{ strong: <TextStrong /> }} />
                    </li>
                </ul>
                <p>
                    <Trans i18nKey="screens.home.what.outro" components={{ em: <em /> }} />
                </p>
            </Section>

            <Section
                title={t('screens.home.built.title')}
                imageSrc="/Race Lineup.png"
                imageAlt={t('screens.home.images.builtAlt')}
                imageSide="left"
            >
                <p>{t('screens.home.built.intro')}</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>{t('screens.home.built.bullets.distance')}</li>
                    <li>{t('screens.home.built.bullets.multiple')}</li>
                    <li>{t('screens.home.built.bullets.conditions')}</li>
                </ul>
                <p>{t('screens.home.built.outro')}</p>
                <Text as="p" variant="support">
                    <Trans i18nKey="screens.home.built.notDesigned" components={{ strong: <TextStrong /> }} />
                </Text>
            </Section>

            <Section title={t('screens.home.how.title')}>
                <HowStepStrip stepKey={howStepKey} onStepKeyChange={setHowStepKey} />
                <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <Text as="div">{t(`screens.home.how.steps.${howStepKey}.body`)}</Text>
                </div>
                <Text as="div" variant="muted" className="mt-3">
                    {t('screens.home.how.tip')}
                </Text>
            </Section>

            <Section
                title={t('screens.home.who.title')}
                imageSrc="/Clinic.webp"
                imageAlt={t('screens.home.images.whoAlt')}
                imageSide="right"
            >
                <ul className="list-disc pl-5 space-y-1">
                    <li>{t('screens.home.who.bullets.windsurfers')}</li>
                    <li>{t('screens.home.who.bullets.coaches')}</li>
                    <li>{t('screens.home.who.bullets.friends')}</li>
                    <li>{t('screens.home.who.bullets.anyone')}</li>
                </ul>
                <p>{t('screens.home.who.outro')}</p>
            </Section>

            <Section title={t('screens.home.why.title')}>
                <p>{t('screens.home.why.intro')}</p>
                <Text as="p" variant="support">
                    {t('screens.home.why.outro')}
                </Text>
            </Section>

            <GetStartedSection />

            <SupportProjectSection />
        </div>
    )
}

const Section: React.FC<{
    title: string
    children: React.ReactNode
    imageSrc?: string
    imageAlt?: string
    imageSide?: 'left' | 'right'
}> = ({ title, children, imageSrc, imageAlt, imageSide = 'right' }) => {
    const hasImage = !!imageSrc && !!imageAlt
    const isImageLeft = imageSide === 'left'
    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
            {hasImage ? (
                <div className="relative">
                    <div className={isImageLeft ? 'lg:pl-[calc(45%+1.5rem)]' : 'lg:pr-[calc(45%+1.5rem)]'}>
                        <Heading level={2}>{title}</Heading>
                        <TextStack>{children}</TextStack>
                    </div>

                    <div
                        className={[
                            'hidden lg:block absolute top-0 bottom-0 w-[45%]',
                            isImageLeft ? 'left-0' : 'right-0',
                        ].join(' ')}
                    >
                        <div className="h-full rounded-xl overflow-hidden border border-slate-200 bg-slate-100">
                            <img
                                src={imageSrc}
                                alt={imageAlt}
                                className="h-full w-full object-cover object-center"
                                loading="lazy"
                            />
                        </div>
                    </div>

                    <div className="lg:hidden mt-4 rounded-xl overflow-hidden border border-slate-200 bg-slate-100">
                        <img
                            src={imageSrc}
                            alt={imageAlt}
                            className="block w-full max-h-72 object-cover object-center"
                            loading="lazy"
                        />
                    </div>
                </div>
            ) : (
                <>
                    <Heading level={2}>{title}</Heading>
                    <TextStack>{children}</TextStack>
                </>
            )}
        </section>
    )
}

const HowStepStrip: React.FC<{
    stepKey: 'ingress' | 'drop' | 'process' | 'review'
    onStepKeyChange: (k: 'ingress' | 'drop' | 'process' | 'review') => void
}> = ({ stepKey, onStepKeyChange }) => {
    const { t } = useTranslation()
    const steps = [
        {
            key: 'ingress',
            label: <Trans i18nKey="screens.home.how.steps.ingress.title" components={{ strong: <TextStrong /> }} />,
        },
        {
            key: 'drop',
            label: <Trans i18nKey="screens.home.how.steps.drop.title" components={{ strong: <TextStrong /> }} />,
        },
        {
            key: 'process',
            label: <Trans i18nKey="screens.home.how.steps.process.title" components={{ strong: <TextStrong /> }} />,
        },
        {
            key: 'review',
            label: <Trans i18nKey="screens.home.how.steps.review.title" components={{ strong: <TextStrong /> }} />,
        },
    ] as const

    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
            {steps.map((s, idx) => {
                const active = s.key === stepKey
                return (
                    <button
                        key={s.key}
                        type="button"
                        onClick={() => onStepKeyChange(s.key)}
                        onMouseEnter={() => onStepKeyChange(s.key)}
                        onFocus={() => onStepKeyChange(s.key)}
                        className={[
                            'text-left rounded-xl border px-3 py-3 transition',
                            active
                                ? 'border-brand-600/30 bg-brand-50'
                                : 'border-slate-200 bg-white hover:border-slate-300',
                        ].join(' ')}
                    >
                        <div className="text-[11px] font-semibold text-slate-500">
                            {t('screens.home.how.stepLabel', { n: idx + 1 })}
                        </div>
                        <div className="mt-1 text-sm font-semibold text-slate-900">{s.label}</div>
                    </button>
                )
            })}
        </div>
    )
}
