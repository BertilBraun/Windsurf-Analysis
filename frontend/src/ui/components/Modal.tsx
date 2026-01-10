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
    closeOnBackdropClick?: boolean
    closeOnEscape?: boolean
    showCloseButton?: boolean
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
    closeOnBackdropClick = true,
    closeOnEscape = true,
    showCloseButton = true,
}) => {
    const { t } = useTranslation()
    const defaultContent = 'rounded-2xl border border-slate-200 bg-white shadow-xl'
    const contentClasses = contentClassName ?? defaultContent
    const contentRef = React.useRef<HTMLDivElement | null>(null)
    React.useEffect(() => {
        contentRef.current?.focus?.()
    }, [])

    React.useEffect(() => {
        if (!onClose) return
        if (!closeOnEscape) return
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key !== 'Escape') return
            e.preventDefault()
            onClose()
        }
        document.addEventListener('keydown', onKeyDown)
        return () => document.removeEventListener('keydown', onKeyDown)
    }, [closeOnEscape, onClose])

    return (
        <div
            className={`fixed inset-0 z-50 flex items-center justify-center ${containerClassName || ''}`}
            role="dialog"
            aria-modal="true"
        >
            <div
                className={`absolute inset-0 ${backdropClassName || 'bg-black/60'}`}
                onMouseDown={e => {
                    // Prevent "click-through" into the underlying app (e.g. collapsing panels on document mousedown).
                    e.preventDefault()
                    e.stopPropagation()
                    if (closeOnBackdropClick) onClose?.()
                }}
                onClick={e => {
                    // Extra safety: stop any bubbling click handlers even if closeOnBackdropClick is false.
                    e.preventDefault()
                    e.stopPropagation()
                }}
            />
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
                        {onClose && showCloseButton && (
                            <Button size="sm" variant="ghost" onClick={onClose} text={t('components.modal.close')} />
                        )}
                    </div>
                )}
                {children}
            </div>
        </div>
    )
}
