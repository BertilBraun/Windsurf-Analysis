import React from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'

function cx(...parts: Array<string | undefined | null | false>) {
    return parts.filter(Boolean).join(' ')
}

type BaseProps = {
    className?: string
    imgClassName?: string
    imgStyle?: React.CSSProperties
    'aria-label'?: string
    title?: string
    src?: string
    alt?: string
}

type LinkVariantProps = {
    to: string
    onClick?: never
    type?: never
}

type ButtonVariantProps = {
    to?: never
    onClick: () => void
    type?: React.ButtonHTMLAttributes<HTMLButtonElement>['type']
}

export type LogoButtonProps = BaseProps & (LinkVariantProps | ButtonVariantProps)

export const LogoButton: React.FC<LogoButtonProps> = ({
    className,
    imgClassName,
    imgStyle,
    title,
    src = '/logo.png',
    alt,
    ...props
}) => {
    const { t } = useTranslation()
    const commonClassName = cx('flex items-center gap-2 rounded-md px-2 py-1 hover:bg-slate-100', className)
    const resolvedTitle = title ?? t('components.logoButton.title')
    const resolvedAlt = alt ?? t('components.logoButton.alt')
    const ariaLabel = props['aria-label'] ?? t('components.logoButton.ariaLabel')

    const img = (
        <img src={src} alt={resolvedAlt} className={cx('h-7 w-auto', imgClassName)} style={imgStyle} />
    )

    if ('to' in props) {
        return (
            <Link to={props.to} className={commonClassName} aria-label={ariaLabel} title={resolvedTitle}>
                {img}
            </Link>
        )
    }

    return (
        <button
            type={props.type ?? 'button'}
            onClick={props.onClick}
            className={commonClassName}
            aria-label={ariaLabel}
            title={resolvedTitle}
        >
            {img}
        </button>
    )
}

