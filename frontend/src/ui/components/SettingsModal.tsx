/**
 * @module SettingsModal
 * @description Provides a modal interface for user settings, including language selection,
 * logout functionality, and account deletion.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { Modal } from './Modal'
import { Button } from './Button'
import { LanguageSwitcher } from './LanguageSwitcher'
import { useSettings } from '../hooks/useSettings'
import { UploadQuality } from '../types'
import { useAuth } from '../auth/AuthProvider'

/**
 * A modal component that displays user settings and account management actions.
 *
 * @param props - Component properties.
 * @param props.onClose - Callback function to close the modal.
 * @param props.onLogout - Callback function to handle user logout.
 */
export const SettingsModal: React.FC<{
    onClose: () => void
    onLogout: () => void
}> = ({ onClose, onLogout }) => {
    const { t } = useTranslation()
    // const { settings, setUploadQuality } = useSettings()
    const { uid, authorizedFetch } = useAuth()
    const [isDeleting, setIsDeleting] = React.useState(false)
    const [deleteError, setDeleteError] = React.useState<string | null>(null)

    return (
        <Modal onClose={onClose} title={t('components.settingsModal.title')}>
            <div className="p-4 space-y-4">
                {/*<div>
                    <label className="block text-sm font-medium text-slate-900 mb-1">Upload quality</label>
                    <select
                        className="w-full bg-white border border-slate-200 rounded-md p-2 text-slate-900"
                        value={settings.uploadQuality}
                        onChange={e => setUploadQuality(e.target.value as UploadQuality)}
                    >
                        <option value="original">Original size</option>
                        <option value="high">High quality</option>
                        <option value="medium">Medium quality (recommended)</option>
                        <option value="minimum">Minimum size</option>
                    </select>
                    <p className="mt-2 text-sm text-slate-600">
                        Higher quality improves detections but increases upload time and bandwidth.
                    </p>
                </div>*/}

                <div className="flex flex-row gap-2">
                    <div className="text-sm font-medium text-slate-900 mb-2">
                        {t('components.languageSwitcher.menuLabel')}
                    </div>
                    <LanguageSwitcher />
                </div>

                <div className="pt-2 border-t border-slate-200">
                    <div className="flex flex-col gap-2">
                        <Button variant="danger" onClick={onLogout} text={t('components.settingsModal.logout')} />

                        <Button
                            variant="danger"
                            disabled={!uid || isDeleting}
                            onClick={async () => {
                                if (!uid) return
                                const ok = window.confirm(t('components.settingsModal.confirmDelete'))
                                if (!ok) return

                                setDeleteError(null)
                                setIsDeleting(true)
                                try {
                                    const res = await authorizedFetch(`/users/${uid}`, { method: 'DELETE' })
                                    if (!res.ok) throw new Error(await res.text())
                                    onLogout()
                                    onClose()
                                } catch (e: any) {
                                    setDeleteError(e?.message || String(e))
                                }
                                setIsDeleting(false)
                            }}
                            text={
                                isDeleting
                                    ? t('components.settingsModal.deletingAccount')
                                    : t('components.settingsModal.deleteAccount')
                            }
                        />

                        {deleteError && <div className="text-sm text-red-700">{deleteError}</div>}
                    </div>
                </div>
            </div>
        </Modal>
    )
}
