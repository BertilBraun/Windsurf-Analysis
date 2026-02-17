/**
 * @file KeyboardShortcutsModal.tsx
 * @description Modal component that displays a list of available keyboard shortcuts for the application.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { Modal } from './Modal'

/**
 * A modal component that displays a categorized list of keyboard shortcuts and their descriptions.
 *
 * @param props - Component properties.
 * @param props.onClose - Callback function invoked when the modal is requested to close.
 */
export const KeyboardShortcutsModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
    const { t } = useTranslation()
    return (
        <Modal onClose={onClose} title={t('components.keyboardShortcutsModal.title')}>
            <div className="p-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <Shortcut keys={['Space']} desc={t('components.keyboardShortcutsModal.shortcuts.playPause')} />
                    <Shortcut keys={['Esc']} desc={t('components.keyboardShortcutsModal.shortcuts.back')} />
                    <Shortcut keys={['Arrow Left']} desc={t('components.keyboardShortcutsModal.shortcuts.prevFrame')} />
                    <Shortcut keys={['Arrow Right']} desc={t('components.keyboardShortcutsModal.shortcuts.nextFrame')} />
                    <Shortcut keys={['Shift', 'Arrow Left']} desc={t('components.keyboardShortcutsModal.shortcuts.seekMinus5')} />
                    <Shortcut keys={['Shift', 'Arrow Right']} desc={t('components.keyboardShortcutsModal.shortcuts.seekPlus5')} />
                    <Shortcut keys={['Ctrl', 'Arrow Left']} desc={t('components.keyboardShortcutsModal.shortcuts.seekMinus30')} />
                    <Shortcut keys={['Ctrl', 'Arrow Right']} desc={t('components.keyboardShortcutsModal.shortcuts.seekPlus30')} />
                    <Shortcut keys={['-']} desc={t('components.keyboardShortcutsModal.shortcuts.slowDown')} />
                    <Shortcut keys={['+']} desc={t('components.keyboardShortcutsModal.shortcuts.speedUp')} />
                    <Shortcut keys={['N']} desc={t('components.keyboardShortcutsModal.shortcuts.nextTrack')} />
                    <Shortcut keys={['P']} desc={t('components.keyboardShortcutsModal.shortcuts.prevTrack')} />
                    <Shortcut keys={['Shift', 'N']} desc={t('components.keyboardShortcutsModal.shortcuts.nextVideo')} />
                    <Shortcut keys={['Shift', 'P']} desc={t('components.keyboardShortcutsModal.shortcuts.prevVideo')} />
                    <Shortcut keys={['Shift', 'Delete']} desc={t('components.keyboardShortcutsModal.shortcuts.deleteFile')} />
                    <Shortcut keys={['Ctrl', 'Z']} desc={t('components.keyboardShortcutsModal.shortcuts.undoDraw')} />
                    <Shortcut keys={['D']} desc={t('components.keyboardShortcutsModal.shortcuts.toggleDraw')} />
                </div>
                <div className="mt-4 text-sm text-slate-600">{t('components.keyboardShortcutsModal.tip')}</div>
            </div>
        </Modal>
    )
}

const Shortcut: React.FC<{ keys: string[]; desc: string }> = ({ keys, desc }) => {
    return (
        <div className="flex items-center justify-between gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
            <div className="text-sm text-slate-700">{desc}</div>
            <div className="flex flex-wrap items-center gap-1">
                {keys.map((k, idx) => (
                    <kbd
                        key={`${k}-${idx}`}
                        className="rounded border border-brand-300/70 bg-brand-50 px-2 py-1 text-xs font-mono text-brand-900"
                    >
                        {k}
                    </kbd>
                ))}
            </div>
        </div>
    )
}
