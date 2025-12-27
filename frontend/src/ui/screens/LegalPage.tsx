/** NOTE: Under no circumstances should you translate this page. It is only used for legal purposes. */

import React from 'react'
import { Link } from 'react-router-dom'
import { getAnalyticsConsent, initAnalytics, installClickTracking, setAnalyticsConsent } from '../utils/analytics'
import { Button } from '../components/Button'
import { Heading, Text, TextStack } from '../components/Typography'

const OWNER_NAME = 'Bertil Braun'
const OWNER_ADDRESS_LINES = ['Im Rübländer 19', '71034 Böblingen', 'Germany']
const OWNER_EMAIL = 'contact@gybelock.de'
const SERVICE_URL = 'https://gybelock.de'

const LAST_UPDATED = '2025-12-21'

function ExternalLink({ href, children }: { href: string; children: React.ReactNode }) {
    return (
        <a className="text-brand-700 underline underline-offset-4" href={href} target="_blank" rel="noreferrer">
            {children}
        </a>
    )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <section className="space-y-3 pt-8">
            <Heading level={2}>{title}</Heading>
            <TextStack>{children}</TextStack>
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
    const title =
        kind === 'terms'
            ? 'Terms of Use'
            : kind === 'privacy'
            ? 'Privacy Policy'
            : kind === 'impressum'
            ? 'Impressum'
            : 'Contact'

    return (
        <div className="max-w-3xl">
            <Heading level={1} className="mb-2">
                {title}
            </Heading>
            <Text as="div" variant="muted" className="mb-6">
                Last updated: <span className="font-medium text-slate-700">{LAST_UPDATED}</span>
            </Text>

            {kind === 'contact' && <ContactContent />}

            {kind === 'impressum' && <ImpressumContent />}

            {kind === 'terms' && <TermsContent />}

            {kind === 'privacy' && <PrivacyContent />}
        </div>
    )
}

const ContactContent: React.FC = () => {
    return (
        <TextStack>
            <p>
                For help, feedback, or data protection requests, email{' '}
                <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`}>
                    {OWNER_EMAIL}
                </a>
            </p>
        </TextStack>
    )
}

const ImpressumContent: React.FC = () => {
    return (
        <TextStack className="space-y-6">
            <p>
                <Pill>Germany</Pill> <Pill>Private individual</Pill> <Pill>No company</Pill>
            </p>

            <Section title="Provider information (Angaben gemäß § 5 DDG)">
                <div className="text-sm text-slate-700">
                    <div className="font-semibold">{OWNER_NAME}</div>
                    <div>{OWNER_ADDRESS_LINES[0]}</div>
                    <div>{OWNER_ADDRESS_LINES[1]}</div>
                    <div>{OWNER_ADDRESS_LINES[2]}</div>
                    <div className="mt-2">
                        Email:{' '}
                        <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`}>
                            {OWNER_EMAIL}
                        </a>
                    </div>
                </div>
            </Section>

            <Section title="Responsible for content (Verantwortlich i.S.d. § 18 Abs. 2 MStV)">
                <p>
                    {OWNER_NAME}, {OWNER_ADDRESS_LINES.join(', ')}
                </p>
            </Section>

            <Section title="Dispute resolution / Verbraucherstreitbeilegung">
                <p>
                    I am not willing or obliged to participate in dispute resolution proceedings before a consumer
                    arbitration board.
                </p>
            </Section>

            <Section title="Liability for contents and links (Haftung für Inhalte & Links)">
                <p>
                    I make every effort to keep the information on this site up to date, but I do not assume liability
                    for the correctness, completeness, or timeliness of contents. This site may contain links to
                    external websites; I have no influence on their content and therefore cannot assume any liability
                    for them.
                </p>
            </Section>
        </TextStack>
    )
}

const TermsContent: React.FC = () => {
    return (
        <TextStack className="space-y-6">
            <p>
                These Terms of Use govern your use of <b>GybeLock</b> (the “Service”) available at{' '}
                <ExternalLink href={SERVICE_URL}>{SERVICE_URL.replace(/^https?:\/\//, '')}</ExternalLink>. By creating
                an account or using the Service, you agree to these Terms.
            </p>

            <Section title="1. Who we are">
                <p>
                    The Service is provided by <b>{OWNER_NAME}</b>, {OWNER_ADDRESS_LINES.join(', ')} (
                    <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`}>
                        {OWNER_EMAIL}
                    </a>
                    ).
                </p>
                <p>
                    Legal notice: see{' '}
                    <Link className="text-brand-700 underline underline-offset-4" to="/impressum">
                        Impressum
                    </Link>
                    .
                </p>
            </Section>

            <Section title="2. What GybeLock does">
                <p>
                    GybeLock is a windsurf-session review tool focused on finding and reviewing gybes/jibes (or other
                    moments where a rider is visible) by automatically detecting and tracking surfers in your uploaded
                    videos.
                </p>
                <p>
                    Typical workflow: you record a session (shore / beach tele footage), drop raw MP4 files into a
                    designated folder, GybeLock uploads and processes them, and you review the finished results by
                    jumping between tracked rider segments.
                </p>
                <p>This is a small-scale, donation-supported project and may be offered as a beta service.</p>
            </Section>

            <Section title="3. Account, eligibility">
                <ul className="list-disc pl-5 space-y-1">
                    <li>You need an account (email/password or Google sign-in) to use the Analyzer.</li>
                    <li>
                        If you use email/password sign-up, you must verify your email address before using the backend
                        processing features.
                    </li>
                    <li>If you are under 16, you may only use the Service with consent of your legal guardian.</li>
                </ul>
            </Section>

            <Section title="4. Your content (uploaded videos)">
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        You keep all rights to your videos. You grant the provider a limited right to process your
                        uploads solely for providing the Service.
                    </li>
                    <li>
                        <b>Video storage</b>: uploaded videos are processed and then deleted from the server side. The
                        Service keeps the generated tracking results so you can review them in the Analyzer.
                    </li>
                    <li>
                        You are responsible for the content you upload and must have the necessary rights (and
                        permissions) to upload and process the videos.
                    </li>
                    <li>
                        Do not upload illegal content or content that infringes rights of others (e.g., privacy,
                        copyright).
                    </li>
                </ul>
            </Section>

            <Section title="5. Acceptable use">
                <ul className="list-disc pl-5 space-y-1">
                    <li>No attempts to overload or disrupt the Service (e.g., abusive automation).</li>
                    <li>No reverse engineering of security mechanisms or bypassing quotas/limits.</li>
                    <li>No use for unlawful purposes.</li>
                </ul>
            </Section>

            <Section title="6. Availability, changes">
                <p>
                    The Service is provided “as is”. I may modify, limit, or discontinue parts of the Service at any
                    time (e.g., to improve performance, fix bugs, or manage costs).
                </p>
            </Section>

            <Section title="7. Liability">
                <p>
                    The Service is provided free of charge (donations optional). I am liable without limitation for
                    intent and gross negligence and for damages resulting from injury to life, body, or health. In cases
                    of slight negligence, I am only liable for breach of essential contractual obligations (cardinal
                    duties), and liability is limited to typical, foreseeable damages.
                </p>
                <p>Mandatory statutory liability (e.g., under product liability law) remains unaffected.</p>
            </Section>

            <Section title="8. Termination, deletion">
                <p>
                    You can delete your account from the Analyzer settings. Deletion impacts data as described in the{' '}
                    <Link className="text-brand-700 underline underline-offset-4" to="/privacy">
                        Privacy Policy
                    </Link>
                    .
                </p>
            </Section>

            <Section title="9. Governing law">
                <p>
                    German law applies. If you are a consumer, mandatory consumer protection rules of your country of
                    residence remain unaffected.
                </p>
            </Section>

            <Section title="10. Contact">
                <p>
                    Questions about these Terms:{' '}
                    <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`}>
                        {OWNER_EMAIL}
                    </a>
                </p>
            </Section>
        </TextStack>
    )
}

const PrivacyContent: React.FC = () => {
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

    return (
        <TextStack className="space-y-6">
            <p>
                This Privacy Policy explains how <b>GybeLock</b> processes personal data when you use the Service at{' '}
                <ExternalLink href={SERVICE_URL}>{SERVICE_URL.replace(/^https?:\/\//, '')}</ExternalLink>.
            </p>

            <Section title="1. Controller (Verantwortlicher)">
                <p>
                    <b>{OWNER_NAME}</b>
                    <br />
                    {OWNER_ADDRESS_LINES.join(', ')}
                    <br />
                    Email:{' '}
                    <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`}>
                        {OWNER_EMAIL}
                    </a>
                </p>
            </Section>

            <Section title="2. What data we process">
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <b>Account data</b>: email address, Firebase user ID (UID), email verification status.
                    </li>
                    <li>
                        <b>Contact preferences</b>: your consent choice for feedback/marketing emails and the timestamp
                        of that consent (if given).
                    </li>
                    <li>
                        <b>Uploaded video data (temporary)</b>: video files you upload for processing; file name, MIME
                        type, file size; checksums (e.g., SHA-256) used for deduplication. Videos are only processed and
                        then deleted; they are not stored long-term.
                    </li>
                    <li>
                        <b>Job / processing data</b>: job IDs, timestamps, processing status, error messages, and the
                        generated tracking results (e.g., tracks/segments) needed to show you analysis.
                    </li>
                    <li>
                        <b>Technical logs</b>: IP address and request logs may be processed by our hosting/providers to
                        operate and secure the Service.
                    </li>
                    <li>
                        <b>Usage analytics (optional)</b>: if you consent, Google Analytics collects page views and
                        click events (e.g., “ui_click”, “analysis_upload_start”).
                    </li>
                </ul>
                <p className="text-xs text-slate-500">
                    Note: GybeLock also uses a local “ingress folder” on your device. Selecting a folder and playing a
                    local file happens on your device; uploads only occur when you add videos and the Analyzer uploads
                    them for processing.
                </p>
                <p className="text-xs text-slate-500">
                    The Service also stores some settings locally in your browser (e.g., your analytics consent choice).
                </p>
            </Section>

            <Section title="3. Purposes and legal bases (GDPR Art. 6)">
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <b>Provide the Service</b> (account login, uploads, processing, showing results): Art. 6(1)(b)
                        GDPR (performance of a contract).
                    </li>
                    <li>
                        <b>Security, abuse prevention, reliability</b> (rate limiting, dedupe, debugging): Art. 6(1)(f)
                        GDPR (legitimate interests).
                    </li>
                    <li>
                        <b>Usage analytics</b> (Google Analytics): Art. 6(1)(a) GDPR (consent). You can withdraw consent
                        at any time (see “Analytics consent” below).
                    </li>
                    <li>
                        <b>Feedback/marketing emails</b> (optional): Art. 6(1)(a) GDPR (consent). You can withdraw
                        consent at any time by contacting us.
                    </li>
                    <li>
                        <b>Donations link</b> (PayPal): when you click the PayPal link you leave our site; processing by
                        PayPal is governed by PayPal’s policies.
                    </li>
                </ul>
            </Section>

            <Section title="4. Analytics consent (Google Analytics)">
                <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="text-sm text-slate-700">
                        Current choice:{' '}
                        <b>{consent === 'accepted' ? 'Accepted' : consent === 'declined' ? 'Declined' : 'Not set'}</b>
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => setChoice('declined')}
                            text="Decline"
                        />
                        <Button
                            type="button"
                            variant="primary"
                            size="sm"
                            onClick={() => setChoice('accepted')}
                            text="Accept"
                        />
                    </div>
                    <div className="mt-2 text-xs text-slate-500">
                        If declined, GybeLock will not load Google Analytics scripts.
                    </div>
                </div>
            </Section>

            <Section title="5. Recipients / service providers">
                <p>The Service uses the following providers:</p>
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <b>Google (Firebase)</b>: authentication and database (Firebase Authentication, Firestore).
                    </li>
                    <li>
                        <b>Google Analytics</b>: only if you consent (see above).
                    </li>
                    <li>
                        <b>Google Cloud Run</b>: backend API hosting (regional endpoint configured by the operator).
                    </li>
                    <li>
                        <b>Modal</b>: GPU processing infrastructure for the analysis pipeline.
                    </li>
                    <li>
                        <b>PayPal</b>: only if you click the donation link (PayPal acts as its own controller).
                    </li>
                </ul>
            </Section>

            <Section title="6. International transfers">
                <p>
                    Some providers (e.g., Google, Modal) may process data outside the EU/EEA (e.g., in the United
                    States). Where required, transfers are based on appropriate safeguards such as Standard Contractual
                    Clauses (SCCs) and/or adequacy decisions.
                </p>
            </Section>

            <Section title="7. Retention">
                <ul className="list-disc pl-5 space-y-1">
                    <li>
                        <b>Account data</b> is kept while your account is active.
                    </li>
                    <li>
                        <b>Uploaded videos</b> are processed and then deleted (no long-term storage).
                    </li>
                    <li>
                        <b>Tracking results / job data</b> are stored while your account is active or until you delete
                        the job.
                    </li>
                    <li>
                        <b>Analytics data</b> (if enabled) is retained according to Google Analytics settings.
                    </li>
                    <li>
                        <b>Contact preferences</b> are kept until you withdraw consent or delete your account.
                    </li>
                </ul>
            </Section>

            <Section title="8. Your rights">
                <p>
                    Under the GDPR you may have rights of access, rectification, erasure, restriction, data portability,
                    and objection. You can also withdraw consent (where applicable) at any time.
                </p>
                <p>
                    You can delete your account from the Analyzer settings. For other requests, contact{' '}
                    <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`}>
                        {OWNER_EMAIL}
                    </a>
                    .
                </p>
            </Section>

            <Section title="9. Supervisory authority">
                <p>
                    If you believe your data is processed unlawfully, you can lodge a complaint with a supervisory
                    authority. For Baden-Württemberg (Germany), see{' '}
                    <ExternalLink href="https://www.baden-wuerttemberg.datenschutz.de/">
                        LfDI Baden-Württemberg
                    </ExternalLink>
                    .
                </p>
            </Section>

            <Section title="10. Contact">
                <p>
                    Data protection inquiries:{' '}
                    <a className="text-brand-700 underline underline-offset-4" href={`mailto:${OWNER_EMAIL}`}>
                        {OWNER_EMAIL}
                    </a>
                </p>
            </Section>
        </TextStack>
    )
}
