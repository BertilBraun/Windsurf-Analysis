import React from 'react'
import { useTranslation } from 'react-i18next'
import { Button } from './Button'

export type ModalProps = {
    onClose?: () => void
    children: React.ReactNode
    additionalHeader?: React.ReactNode
    containerClassName?: string
    backdropClassName?: string
    contentClassName?: string
    title?: string
    headerClassName?: string
    hideHeader?: boolean
}

export const Modal: React.FC<ModalProps> = ({
    onClose,
    children,
    additionalHeader,
    containerClassName,
    backdropClassName,
    contentClassName,
    title,
    headerClassName,
    hideHeader,
}) => {
    const { t } = useTranslation()
    const defaultContent = 'rounded-2xl border border-slate-200 bg-white shadow-xl'
    const contentClasses = contentClassName ?? defaultContent
    const contentRef = React.useRef<HTMLDivElement | null>(null)
    React.useEffect(() => {
        contentRef.current?.focus?.()
    }, [])
    return (
        <div
            className={`fixed inset-0 z-50 flex items-center justify-center ${containerClassName || ''}`}
            role="dialog"
            aria-modal="true"
        >
            <div className={`absolute inset-0 ${backdropClassName || 'bg-black/60'}`} onClick={onClose} />
            <div
                className={`relative z-10 ${contentClasses}`}
                onClick={e => e.stopPropagation()}
                tabIndex={-1}
                ref={contentRef}
            >
                {!hideHeader && (title || onClose || additionalHeader) && (
                    <div
                        className={`flex items-center justify-between px-4 py-3 border-b border-slate-200 bg-white/80 backdrop-blur rounded-t-2xl gap-2 ${
                            headerClassName || ''
                        }`}
                    >
                        <div className="font-semibold text-slate-900">{title}</div>
                        <div className="flex-1" />
                        {additionalHeader}
                        {onClose && (
                            <Button size="sm" variant="ghost" onClick={onClose} text={t('components.modal.close')} />
                        )}
                    </div>
                )}
                {children}
            </div>
        </div>
    )
}
