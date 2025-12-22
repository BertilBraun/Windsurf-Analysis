import React from 'react'
import { Trans, useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { Button } from './Button'
import { Modal } from './Modal'

type ConsentModalProps = {
    isSubmitting: boolean
    onSubmit: (marketingConsent: boolean) => void
}

export const ConsentModal: React.FC<ConsentModalProps> = ({ isSubmitting, onSubmit }) => {
    const { t } = useTranslation()
    const [termsAccepted, setTermsAccepted] = React.useState(false)
    const [marketingConsent, setMarketingConsent] = React.useState(false)

    return (
        <Modal title={t('components.consentModal.title')} contentClassName="rounded-2xl border border-slate-200 bg-white shadow-xl w-[520px] max-w-[92vw]">
            <div className="p-5 space-y-4">
                <div className="text-sm text-slate-600">{t('components.consentModal.subtitle')}</div>
                <label className="flex items-start gap-2 text-xs text-slate-700">
                    <input
                        type="checkbox"
                        className="mt-0.5 h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-600/30"
                        checked={termsAccepted}
                        onChange={e => setTermsAccepted(e.target.checked)}
                    />
                    <span>
                        <Trans
                            i18nKey="components.consentModal.terms"
                            components={{
                                termsLink: (
                                    <Link className="text-brand-700 underline underline-offset-2" to="/terms" />
                                ),
                                privacyLink: (
                                    <Link className="text-brand-700 underline underline-offset-2" to="/privacy" />
                                ),
                            }}
                        />
                    </span>
                </label>
                <label className="flex items-start gap-2 text-xs text-slate-700">
                    <input
                        type="checkbox"
                        className="mt-0.5 h-4 w-4 rounded border-slate-300 text-brand-600 focus:ring-brand-600/30"
                        checked={marketingConsent}
                        onChange={e => setMarketingConsent(e.target.checked)}
                    />
                    <span>{t('components.consentModal.marketing')}</span>
                </label>
                <div className="flex justify-end">
                    <Button
                        variant="primary"
                        disabled={!termsAccepted || isSubmitting}
                        isPending={isSubmitting}
                        onClick={() => onSubmit(marketingConsent)}
                        text={t('components.consentModal.accept')}
                    />
                </div>
            </div>
        </Modal>
    )
}
