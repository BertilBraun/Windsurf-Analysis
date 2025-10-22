import React from 'react'
import { Modal } from './Modal'

export const KeyboardShortcutsModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
    return (
        <Modal onClose={onClose} title="Keyboard Shortcuts">
            <div className="p-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <Shortcut keys={['Space']} desc="Play/Pause (restart if at end)" />
                    <Shortcut keys={['Esc']} desc="Back: exit detailed; in overview close player" />
                    <Shortcut keys={['Arrow Left']} desc="Previous frame" />
                    <Shortcut keys={['Arrow Right']} desc="Next frame" />
                    <Shortcut keys={['Shift', 'Arrow Left']} desc="Seek -5s" />
                    <Shortcut keys={['Shift', 'Arrow Right']} desc="Seek +5s" />
                    <Shortcut keys={['Ctrl', 'Arrow Left']} desc="Seek -30s" />
                    <Shortcut keys={['Ctrl', 'Arrow Right']} desc="Seek +30s" />
                    <Shortcut keys={['-']} desc="Slow down" />
                    <Shortcut keys={['+']} desc="Speed up" />
                    <Shortcut keys={['N']} desc="Next track" />
                    <Shortcut keys={['P']} desc="Previous track" />
                    <Shortcut keys={['Shift', 'N']} desc="Open next video" />
                    <Shortcut keys={['Shift', 'P']} desc="Open previous video" />
                </div>
                <div className="mt-4 text-sm text-gray-400">Tip: Use the mouse wheel to zoom in overview mode.</div>
            </div>
        </Modal>
    )
}

const Shortcut: React.FC<{ keys: string[]; desc: string }> = ({ keys, desc }) => {
    return (
        <div className="flex items-center justify-between gap-2 rounded-md border border-gray-700 bg-black/40 px-3 py-2">
            <div className="flex flex-wrap items-center gap-1">
                {keys.map((k, idx) => (
                    <kbd
                        key={`${k}-${idx}`}
                        className="rounded border border-gray-600 bg-[#1a1a1a] px-2 py-1 text-xs font-mono text-gray-200"
                    >
                        {k}
                    </kbd>
                ))}
            </div>
            <div className="text-sm text-gray-200">{desc}</div>
        </div>
    )
}
