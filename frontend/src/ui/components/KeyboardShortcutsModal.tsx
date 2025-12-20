import React from 'react'
import { useTranslation } from 'react-i18next'
import { Modal } from './Modal'

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
                    <Shortcut keys={['D']} desc={t('components.keyboardShortcutsModal.shortcuts.toggleDraw')} />
                    <Shortcut keys={['N']} desc={t('components.keyboardShortcutsModal.shortcuts.nextTrack')} />
                    <Shortcut keys={['P']} desc={t('components.keyboardShortcutsModal.shortcuts.prevTrack')} />
                    <Shortcut keys={['Shift', 'N']} desc={t('components.keyboardShortcutsModal.shortcuts.nextVideo')} />
                    <Shortcut keys={['Shift', 'P']} desc={t('components.keyboardShortcutsModal.shortcuts.prevVideo')} />
                </div>
                <div className="mt-4 text-sm text-slate-600">{t('components.keyboardShortcutsModal.tip')}</div>
            </div>
        </Modal>
    )
}

const Shortcut: React.FC<{ keys: string[]; desc: string }> = ({ keys, desc }) => {
    return (
        <div className="flex items-center justify-between gap-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
            <div className="flex flex-wrap items-center gap-1">
                {keys.map((k, idx) => (
                    <kbd
                        key={`${k}-${idx}`}
                        className="rounded border border-slate-200 bg-white px-2 py-1 text-xs font-mono text-slate-700"
                    >
                        {k}
                    </kbd>
                ))}
            </div>
            <div className="text-sm text-slate-700">{desc}</div>
        </div>
    )
}
