import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { getAnalyticsConsent, initAnalytics, installClickTracking, setAnalyticsConsent } from '../utils/analytics'

const OWNER_NAME = 'Bertil Braun'
const OWNER_ADDRESS_LINES = ['Im Raessblaender 19', '71034 Boeblingen', 'Germany']
const OWNER_EMAIL = 'contact@gybelock.de'
const SERVICE_URL = 'https://gybelock.de'

const LAST_UPDATED = '2025-12-18'

function ExternalLink({ href, children }: { href: string; children: React.ReactNode }) {
    return (
        <a className="text-brand-700 underline underline-offset-4" href={href} target="_blank" rel="noreferrer">
            {children}
        </a>
    )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <section className="space-y-3">
            <h2 className="mt-8 mb-0">{title}</h2>
            <div className="text-sm text-slate-600 leading-6 space-y-3">{children}</div>
        </section>
    )
}

function Pill({ children }: { children: React.ReactNode }) {
    return (
        <span className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[11px] text-slate-700">
            {children}
        </span>
    )
}

export const LegalPage: React.FC<{ kind: 'terms' | 'privacy' | 'impressum' | 'contact' }> = ({ kind }) => {
    const { t } = useTranslation()
    const title = t(`screens.legal.titles.${kind}`)

    return (
        <div className="max-w-3xl">
            <h1 className="mb-2">{title}</h1>
            <div className="text-xs text-slate-500 mb-6">
                {t('screens.legal.lastUpdated', { date: LAST_UPDATED })}
            </div>

            {kind === 'contact' && (
                <div className="text-sm text-slate-600 leading-6 space-y-3">
                    <p>
                        <Trans
                            i18nKey="screens.legal.contact.body"
                            components={{
                                emailLink: (
                                    <a
                                        className="text-brand-700 underline underline-offset-4"
                                        href={`mailto:${OWNER_EMAIL}`}
                                    />
                                ),
                            }}
                            values={{ email: OWNER_EMAIL }}
                        />
                    </p>
                </div>
            )}

            {kind === 'impressum' && (
                <div className="text-sm text-slate-600 leading-6 space-y-6">
                    <p className="text-slate-700">
                        <Pill>{t('screens.legal.impressum.pills.country')}</Pill>{' '}
                        <Pill>{t('screens.legal.impressum.pills.private')}</Pill>{' '}
                        <Pill>{t('screens.legal.impressum.pills.noCompany')}</Pill>
                    </p>

                    <Section title={t('screens.legal.impressum.provider.title')}>
                        <div className="text-sm text-slate-700">
                            <div className="font-semibold">{OWNER_NAME}</div>
                            <div>{OWNER_ADDRESS_LINES[0]}</div>
                            <div>{OWNER_ADDRESS_LINES[1]}</div>
                            <div>{OWNER_ADDRESS_LINES[2]}</div>
                            <div className="mt-2">
                                {t('screens.legal.impressum.provider.emailLabel')}{' '}
                                <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`}>
                                    {OWNER_EMAIL}
                                </a>
                            </div>
                        </div>
                    </Section>

                    <Section title={t('screens.legal.impressum.responsible.title')}>
                        <p>
                            {OWNER_NAME}, {OWNER_ADDRESS_LINES.join(', ')}
                        </p>
                    </Section>

                    <Section title={t('screens.legal.impressum.dispute.title')}>
                        <p>{t('screens.legal.impressum.dispute.body')}</p>
                    </Section>

                    <Section title={t('screens.legal.impressum.liability.title')}>
                        <p>{t('screens.legal.impressum.liability.body')}</p>
                    </Section>
                </div>
            )}

            {kind === 'terms' && (
                <div className="text-sm text-slate-600 leading-6 space-y-6">
                    <p className="text-slate-700">
                        <Trans
                            i18nKey="screens.legal.terms.intro"
                            components={{
                                b: <b />,
                                serviceLink: <ExternalLink href={SERVICE_URL} />,
                            }}
                            values={{ serviceUrl: SERVICE_URL.replace(/^https?:\/\//, '') }}
                        />
                    </p>

                    <Section title={t('screens.legal.terms.sections.who.title')}>
                        <p>
                            <Trans
                                i18nKey="screens.legal.terms.sections.who.body1"
                                components={{ b: <b />, emailLink: <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`} /> }}
                                values={{ name: OWNER_NAME, email: OWNER_EMAIL, address: OWNER_ADDRESS_LINES.join(', ') }}
                            />
                        </p>
                        <p>
                            <Trans
                                i18nKey="screens.legal.terms.sections.who.body2"
                                components={{ impressumLink: <Link className="text-brand-700 underline underline-offset-4" to="/impressum" /> }}
                            />
                        </p>
                    </Section>

                    <Section title={t('screens.legal.terms.sections.what.title')}>
                        <p>{t('screens.legal.terms.sections.what.body1')}</p>
                        <p>{t('screens.legal.terms.sections.what.body2')}</p>
                        <p>{t('screens.legal.terms.sections.what.body3')}</p>
                    </Section>

                    <Section title={t('screens.legal.terms.sections.account.title')}>
                        <ul className="list-disc pl-5 space-y-1">
                            <li>{t('screens.legal.terms.sections.account.bullets.account')}</li>
                            <li>{t('screens.legal.terms.sections.account.bullets.verify')}</li>
                            <li>{t('screens.legal.terms.sections.account.bullets.guardian')}</li>
                        </ul>
                    </Section>

                    <Section title={t('screens.legal.terms.sections.content.title')}>
                        <ul className="list-disc pl-5 space-y-1">
                            <li>{t('screens.legal.terms.sections.content.bullets.rights')}</li>
                            <li>
                                <Trans
                                    i18nKey="screens.legal.terms.sections.content.bullets.storage"
                                    components={{ b: <b /> }}
                                />
                            </li>
                            <li>{t('screens.legal.terms.sections.content.bullets.responsible')}</li>
                            <li>{t('screens.legal.terms.sections.content.bullets.illegal')}</li>
                        </ul>
                    </Section>

                    <Section title={t('screens.legal.terms.sections.acceptable.title')}>
                        <ul className="list-disc pl-5 space-y-1">
                            <li>{t('screens.legal.terms.sections.acceptable.bullets.overload')}</li>
                            <li>{t('screens.legal.terms.sections.acceptable.bullets.reverse')}</li>
                            <li>{t('screens.legal.terms.sections.acceptable.bullets.unlawful')}</li>
                        </ul>
                    </Section>

                    <Section title={t('screens.legal.terms.sections.availability.title')}>
                        <p>{t('screens.legal.terms.sections.availability.body')}</p>
                    </Section>

                    <Section title={t('screens.legal.terms.sections.liability.title')}>
                        <p>{t('screens.legal.terms.sections.liability.body1')}</p>
                        <p>{t('screens.legal.terms.sections.liability.body2')}</p>
                    </Section>

                    <Section title={t('screens.legal.terms.sections.termination.title')}>
                        <p>
                            <Trans
                                i18nKey="screens.legal.terms.sections.termination.body"
                                components={{ privacyLink: <Link className="text-brand-700 underline underline-offset-4" to="/privacy" /> }}
                            />
                        </p>
                    </Section>

                    <Section title={t('screens.legal.terms.sections.law.title')}>
                        <p>{t('screens.legal.terms.sections.law.body')}</p>
                    </Section>

                    <Section title={t('screens.legal.terms.sections.contact.title')}>
                        <p>
                            <Trans
                                i18nKey="screens.legal.terms.sections.contact.body"
                                components={{ emailLink: <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`} /> }}
                                values={{ email: OWNER_EMAIL }}
                            />
                        </p>
                    </Section>
                </div>
            )}

            {kind === 'privacy' && <PrivacyContent />}
        </div>
    )
}

const PrivacyContent: React.FC = () => {
    const { t } = useTranslation()
    const [consent, setConsent] = React.useState<'accepted' | 'declined' | null>(null)

    React.useEffect(() => {
        setConsent(getAnalyticsConsent())
    }, [])

    const setChoice = (v: 'accepted' | 'declined') => {
        setAnalyticsConsent(v)
        setConsent(v)
        if (v === 'accepted') {
            initAnalytics()
            installClickTracking()
        }
    }

    const consentLabel =
        consent === 'accepted'
            ? t('screens.legal.privacy.analytics.accepted')
            : consent === 'declined'
            ? t('screens.legal.privacy.analytics.declined')
            : t('screens.legal.privacy.analytics.notSet')

    return (
        <div className="text-sm text-slate-600 leading-6 space-y-6">
            <p className="text-slate-700">
                <Trans
                    i18nKey="screens.legal.privacy.intro"
                    components={{ b: <b />, serviceLink: <ExternalLink href={SERVICE_URL} /> }}
                    values={{ serviceUrl: SERVICE_URL.replace(/^https?:\/\//, '') }}
                />
            </p>

            <Section title={t('screens.legal.privacy.controller.title')}>
                <p>
                    <b>{OWNER_NAME}</b>
                    <br />
                    {OWNER_ADDRESS_LINES.join(', ')}
                    <br />
                    {t('screens.legal.privacy.controller.emailLabel')}{' '}
                    <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`}>
                        {OWNER_EMAIL}
                    </a>
                </p>
            </Section>

            <Section title={t('screens.legal.privacy.data.title')}>
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <Trans i18nKey="screens.legal.privacy.data.bullets.account" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.data.bullets.uploads" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.data.bullets.jobs" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.data.bullets.logs" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.data.bullets.analytics" components={{ b: <b /> }} />
                    </li>
                </ul>
                <p className="text-xs text-slate-500">{t('screens.legal.privacy.data.noteIngress')}</p>
                <p className="text-xs text-slate-500">{t('screens.legal.privacy.data.noteSettings')}</p>
            </Section>

            <Section title={t('screens.legal.privacy.purposes.title')}>
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <Trans i18nKey="screens.legal.privacy.purposes.bullets.service" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.purposes.bullets.security" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.purposes.bullets.analytics" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.purposes.bullets.paypal" components={{ b: <b /> }} />
                    </li>
                </ul>
            </Section>

            <Section title={t('screens.legal.privacy.analytics.title')}>
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="text-sm text-slate-700">
                        {t('screens.legal.privacy.analytics.currentChoice', { choice: consentLabel })}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                        <button
                            type="button"
                            onClick={() => setChoice('declined')}
                            className="rounded-lg border border-slate-200 px-3 py-2 text-xs text-slate-800 hover:bg-slate-50"
                        >
                            {t('components.analyticsConsentBanner.decline')}
                        </button>
                        <button
                            type="button"
                            onClick={() => setChoice('accepted')}
                            className="rounded-lg bg-slate-900 text-white px-3 py-2 text-xs hover:bg-slate-800"
                        >
                            {t('components.analyticsConsentBanner.accept')}
                        </button>
                    </div>
                    <div className="mt-2 text-xs text-slate-500">{t('screens.legal.privacy.analytics.note')}</div>
                </div>
            </Section>

            <Section title={t('screens.legal.privacy.recipients.title')}>
                <p>{t('screens.legal.privacy.recipients.intro')}</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <Trans i18nKey="screens.legal.privacy.recipients.bullets.firebase" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.recipients.bullets.analytics" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.recipients.bullets.cloudRun" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.recipients.bullets.modal" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.recipients.bullets.paypal" components={{ b: <b /> }} />
                    </li>
                </ul>
            </Section>

            <Section title={t('screens.legal.privacy.transfers.title')}>
                <p>{t('screens.legal.privacy.transfers.body')}</p>
            </Section>

            <Section title={t('screens.legal.privacy.retention.title')}>
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <Trans i18nKey="screens.legal.privacy.retention.bullets.account" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.retention.bullets.uploads" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.retention.bullets.tracking" components={{ b: <b /> }} />
                    </li>
                    <li>
                        <Trans i18nKey="screens.legal.privacy.retention.bullets.analytics" components={{ b: <b /> }} />
                    </li>
                </ul>
            </Section>

            <Section title={t('screens.legal.privacy.rights.title')}>
                <p>{t('screens.legal.privacy.rights.body1')}</p>
                <p>
                    <Trans
                        i18nKey="screens.legal.privacy.rights.body2"
                        components={{ emailLink: <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`} /> }}
                        values={{ email: OWNER_EMAIL }}
                    />
                </p>
            </Section>

            <Section title={t('screens.legal.privacy.authority.title')}>
                <p>
                    <Trans
                        i18nKey="screens.legal.privacy.authority.body"
                        components={{ authorityLink: <ExternalLink href="https://www.baden-wuerttemberg.datenschutz.de/" /> }}
                    />
                </p>
            </Section>

            <Section title={t('screens.legal.privacy.contact.title')}>
                <p>
                    <Trans
                        i18nKey="screens.legal.privacy.contact.body"
                        components={{ emailLink: <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`} /> }}
                        values={{ email: OWNER_EMAIL }}
                    />
                </p>
            </Section>
        </div>
    )
}
