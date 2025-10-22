import React from 'react'
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
}) => {
    const defaultContent = 'rounded-md border border-gray-700 bg-[#111] shadow-xl'
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
                {(title || onClose || additionalHeader) && (
                    <div
                        className={`flex items-center justify-between px-4 py-3 border-b border-gray-700 bg-black/60 rounded-t-md gap-2 ${
                            headerClassName || ''
                        }`}
                    >
                        <div className="font-semibold text-gray-100">{title}</div>
                        <div className="flex-1" />
                        {additionalHeader}
                        {onClose && <Button className="text-sm text-black" onClick={onClose} text="Close" />}
                    </div>
                )}
                {children}
            </div>
        </div>
    )
}
