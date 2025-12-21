import React from 'react'
import { useTranslation } from 'react-i18next'
import { Modal } from './Modal'

export const HelpModal: React.FC<{
    onClose: () => void
}> = ({ onClose }) => {
    const { t } = useTranslation()

    return (
        <Modal onClose={onClose} title={t('screens.analyzer.help.title')}>
            <div className="p-4 text-sm text-slate-700">
                {t('screens.analyzer.help.body')}{' '}
                <a className="text-brand-700 underline" href="mailto:contact@gybelock.de">
                    contact@gybelock.de
                </a>
            </div>
        </Modal>
    )
}
