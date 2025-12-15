import React from 'react'
import { Modal } from './Modal'
import { Button } from './Button'
import { useSettings, UploadQuality } from '../hooks/useSettings'

export const SettingsModal: React.FC<{
    onClose: () => void
    onLogout: () => void
}> = ({ onClose, onLogout }) => {
    const { settings, setUploadQuality } = useSettings()

    return (
        <Modal onClose={onClose} title="Settings">
            <div className="p-4 space-y-4">
                <div>
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
                </div>

                <div className="pt-2 border-t border-slate-200">
                    <Button variant="danger" onClick={onLogout} text="Logout" />
                </div>
            </div>
        </Modal>
    )
}
