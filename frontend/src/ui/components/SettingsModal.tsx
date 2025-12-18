import React from 'react'
import { Modal } from './Modal'
import { Button } from './Button'
import { useSettings } from '../hooks/useSettings'
import { UploadQuality } from '../types'
import { useAuth } from '../auth/AuthProvider'

export const SettingsModal: React.FC<{
    onClose: () => void
    onLogout: () => void
}> = ({ onClose, onLogout }) => {
    const { settings, setUploadQuality } = useSettings()
    const { uid, authorizedFetch } = useAuth()
    const [isDeleting, setIsDeleting] = React.useState(false)
    const [deleteError, setDeleteError] = React.useState<string | null>(null)

    return (
        <Modal onClose={onClose} title="Settings">
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

                <div className="pt-2 border-t border-slate-200">
                    <div className="flex flex-col gap-2">
                        <Button variant="danger" onClick={onLogout} text="Logout" />

                        <Button
                            variant="danger"
                            disabled={!uid || isDeleting}
                            onClick={() => {
                                if (!uid) return
                                const ok = window.confirm(
                                    'Delete your account? This will permanently delete your user and job mappings. This cannot be undone.'
                                )
                                if (!ok) return

                                setDeleteError(null)
                                setIsDeleting(true)
                                void (async () => {
                                    try {
                                        const res = await authorizedFetch(`/users/${uid}`, { method: 'DELETE' })
                                        if (!res.ok) throw new Error(await res.text())
                                        onLogout()
                                        onClose()
                                    } catch (e: any) {
                                        setDeleteError(e?.message || String(e))
                                    } finally {
                                        setIsDeleting(false)
                                    }
                                })()
                            }}
                            text={isDeleting ? 'Deleting account…' : 'Delete account'}
                        />

                        {deleteError && <div className="text-sm text-red-700">{deleteError}</div>}
                    </div>
                </div>
            </div>
        </Modal>
    )
}
