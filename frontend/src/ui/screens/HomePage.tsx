import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/Button'
import { SupportProjectSection } from '../components/SupportProjectSection'
import { GetStartedSection } from '../components/GetStartedSection'

export const HomePage: React.FC = () => {
    const { t } = useTranslation()
    const navigate = useNavigate()
    const [howStepKey, setHowStepKey] = React.useState<'ingress' | 'drop' | 'process' | 'review'>('ingress')
    const demoVideoSrc = React.useMemo(() => {
        const videos = [
            { mp4: '/Surfer1.mp4', av1: '/Surfer1.av1.mp4' },
            { mp4: '/Surfer2.mp4', av1: '/Surfer2.av1.mp4' },
        ]
        return videos[Math.floor(Math.random() * videos.length)]
    }, [])
    return (
        <div className="flex flex-col gap-10">
            <section className="rounded-2xl border border-slate-200 bg-linear-to-b from-white to-slate-50 p-5 sm:p-8">
                <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,2.3fr)] gap-6 items-center">
                    <div className="max-w-md">
                        {/*<div className="text-xs font-semibold text-brand-700 mb-2">{t('screens.home.hero.brand')}</div>*/}
                        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">
                            {t('screens.home.hero.title')}
                        </h1>
                        <br />
                        <p className="mt-3 text-sm sm:text-base text-slate-600 leading-6">
                            <Trans i18nKey="screens.home.hero.lede1" components={{ strong: <strong /> }} />
                        </p>
                        <p className="mt-3 text-sm sm:text-base text-slate-600 leading-6">
                            <Trans i18nKey="screens.home.hero.lede2" components={{ strong: <strong /> }} />
                        </p>
                        <p className="mt-3 text-sm sm:text-base text-slate-600 leading-6">
                            {t('screens.home.hero.lede3')}
                        </p>

                        <br />

                        <div className="mt-5 flex flex-col sm:flex-row gap-3 sm:items-center">
                            <Button
                                variant="primary"
                                onClick={() => navigate('/analyzer')}
                                text={t('screens.home.hero.cta')}
                            />
                            <div className="text-sm text-slate-700">{t('screens.home.hero.freeVideos')}</div>
                        </div>
                        <div className="mt-2 text-xs text-slate-500">{t('screens.home.hero.disclaimer')}</div>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white overflow-hidden">
                        <div className="aspect-video bg-slate-900">
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
                <p className="mt-3">{t('screens.home.problem.outro')}</p>
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
                        <Trans i18nKey="screens.home.what.bullets.stabilizes" components={{ strong: <strong /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.home.what.bullets.locksOnto" components={{ strong: <strong /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.home.what.bullets.centered" components={{ strong: <strong /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.home.what.bullets.follows" components={{ strong: <strong /> }} />
                    </li>
                </ul>
                <div className="mt-3 text-slate-600">
                    <Trans i18nKey="screens.home.what.outro" components={{ em: <em /> }} />
                </div>
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
                <p className="mt-3">
                    <Trans i18nKey="screens.home.built.notDesigned" components={{ strong: <strong /> }} />
                </p>
                <p className="mt-3">{t('screens.home.built.outro')}</p>
            </Section>

            <Section title={t('screens.home.how.title')}>
                <HowStepStrip stepKey={howStepKey} onStepKeyChange={setHowStepKey} />
                <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <div className="text-sm text-slate-700">{t(`screens.home.how.steps.${howStepKey}.body`)}</div>
                </div>
                <div className="mt-3 text-xs text-slate-500">{t('screens.home.how.tip')}</div>
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
                <p className="mt-3">{t('screens.home.who.outro')}</p>
            </Section>

            <Section title={t('screens.home.why.title')}>
                <p>{t('screens.home.why.intro')}</p>
                <p className="mt-2">{t('screens.home.why.outro')}</p>
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
                    <div
                        className={[
                            'text-sm text-slate-600 leading-6 space-y-3',
                            isImageLeft ? 'lg:pl-[calc(40%+1.5rem)]' : 'lg:pr-[calc(40%+1.5rem)]',
                        ].join(' ')}
                    >
                        <h2 className="mt-0">{title}</h2>
                        {children}
                    </div>

                    <div
                        className={[
                            'hidden lg:block absolute top-0 bottom-0 w-[40%]',
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
                    <h2 className="mt-0">{title}</h2>
                    <div className="text-sm text-slate-600 leading-6 space-y-3">{children}</div>
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
            label: <Trans i18nKey="screens.home.how.steps.ingress.title" components={{ strong: <strong /> }} />,
        },
        {
            key: 'drop',
            label: <Trans i18nKey="screens.home.how.steps.drop.title" components={{ strong: <strong /> }} />,
        },
        {
            key: 'process',
            label: <Trans i18nKey="screens.home.how.steps.process.title" components={{ strong: <strong /> }} />,
        },
        {
            key: 'review',
            label: <Trans i18nKey="screens.home.how.steps.review.title" components={{ strong: <strong /> }} />,
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
