import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/Button'
import { SupportProjectSection } from '../components/SupportProjectSection'

export const HomePage: React.FC = () => {
    const { t } = useTranslation()
    const navigate = useNavigate()
    const demoVideoSrc = React.useMemo(() => {
        const videos = [
            { mp4: '/Surfer1.mp4', av1: '/Surfer1.av1.mp4' },
            { mp4: '/Surfer2.mp4', av1: '/Surfer2.av1.mp4' },
        ]
        return videos[Math.floor(Math.random() * videos.length)]
    }, [])
    return (
        <div className="flex flex-col gap-10">
            <section className="rounded-2xl border border-slate-200 bg-linear-to-b from-white to-slate-50 p-6 sm:p-10">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
                    <div className="max-w-2xl">
                        <div className="text-xs font-semibold text-brand-700 mb-2">{t('screens.home.hero.brand')}</div>
                        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight">
                            {t('screens.home.hero.title')}
                        </h1>
                        <p className="mt-3 text-sm sm:text-base text-slate-600 leading-6">
                            <Trans i18nKey="screens.home.hero.lede1" components={{ strong: <strong /> }} />
                        </p>
                        <p className="mt-3 text-sm sm:text-base text-slate-600 leading-6">
                            <Trans i18nKey="screens.home.hero.lede2" components={{ strong: <strong /> }} />
                        </p>
                        <p className="mt-3 text-sm sm:text-base text-slate-600 leading-6">
                            {t('screens.home.hero.lede3')}
                        </p>

                        <div className="mt-5 flex flex-col sm:flex-row gap-3 sm:items-center">
                            <Button
                                variant="primary"
                                onClick={() => navigate('/analyzer')}
                                text={t('screens.home.hero.cta')}
                            />
                            <div className="text-sm text-slate-700">{t('screens.home.hero.freeVideos')}</div>
                        </div>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-white overflow-hidden">
                        <div className="px-4 py-3 border-b border-slate-200 grid grid-cols-2 text-sm font-semibold text-slate-700">
                            <div className="text-center">{t('screens.home.comparison.beforeLabel')}</div>
                            <div className="text-center">{t('screens.home.comparison.afterLabel')}</div>
                        </div>
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
                    </div>
                </div>
            </section>

            <Section title={t('screens.home.problem.title')}>
                <p>{t('screens.home.problem.intro')}</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>{t('screens.home.problem.bullets.zoom')}</li>
                    <li>{t('screens.home.problem.bullets.riders')}</li>
                    <li>{t('screens.home.problem.bullets.moments')}</li>
                </ul>
                <p className="mt-3">{t('screens.home.problem.outro')}</p>
            </Section>

            <Section title={t('screens.home.what.title')}>
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

            <Section title={t('screens.home.built.title')}>
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
                <ol className="list-decimal pl-5 space-y-2">
                    <li>
                        <Trans i18nKey="screens.home.how.steps.ingress.title" components={{ strong: <strong /> }} />
                        <div className="text-sm text-slate-600">{t('screens.home.how.steps.ingress.body')}</div>
                    </li>
                    <li>
                        <Trans i18nKey="screens.home.how.steps.drop.title" components={{ strong: <strong /> }} />
                        <div className="text-sm text-slate-600">{t('screens.home.how.steps.drop.body')}</div>
                    </li>
                    <li>
                        <Trans i18nKey="screens.home.how.steps.process.title" components={{ strong: <strong /> }} />
                        <div className="text-sm text-slate-600">{t('screens.home.how.steps.process.body')}</div>
                    </li>
                    <li>
                        <Trans i18nKey="screens.home.how.steps.review.title" components={{ strong: <strong /> }} />
                        <div className="text-sm text-slate-600">{t('screens.home.how.steps.review.body')}</div>
                    </li>
                </ol>
            </Section>

            <Section title={t('screens.home.who.title')}>
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

            <SupportProjectSection />

            <section className="rounded-2xl border border-brand-600/20 bg-brand-50 p-6 sm:p-8 flex flex-col sm:flex-row gap-4 sm:items-center">
                <div className="flex-1">
                    <div className="text-xs font-semibold text-brand-700">{t('screens.home.ctaSection.title')}</div>
                    <div className="mt-1 text-lg font-semibold text-slate-900">
                        {t('screens.home.ctaSection.headline')}
                    </div>
                    <div className="mt-1 text-sm text-slate-700">{t('screens.home.ctaSection.body')}</div>
                </div>
                <Button
                    variant="primary"
                    onClick={() => navigate('/analyzer')}
                    text={t('screens.home.ctaSection.cta')}
                />
            </section>
        </div>
    )
}

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => {
    return (
        <section className="rounded-2xl border border-slate-200 bg-white p-6 sm:p-8">
            <h2 className="mt-0">{title}</h2>
            <div className="text-sm text-slate-600 leading-6 space-y-3">{children}</div>
        </section>
    )
}
