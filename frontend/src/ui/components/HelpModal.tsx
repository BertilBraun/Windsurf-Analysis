/**
 * @file HelpModal.tsx
 * @description Modal component for displaying help and contact information.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { Modal } from './Modal'

/**
 * A modal component that displays help information and support contact details.
 *
 * @param props - Component properties.
 * @param props.onClose - Callback triggered when the modal is closed.
 */
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
