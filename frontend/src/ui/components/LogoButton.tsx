/**
 * @module LogoButton
 * Provides a clickable logo component that can function as either a navigation link or a button.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { Button } from './Button'

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

/**
 * Props for the LogoButton component.
 *
 * This type is a discriminated union requiring either a `to` prop for navigation (rendering a `Link`)
 * or an `onClick` prop for actions (rendering a `Button`).
 */
export type LogoButtonProps = BaseProps & (LinkVariantProps | ButtonVariantProps)

/**
 * A component that renders a logo image wrapped in either a router `Link` or a `Button`.
 *
 * It automatically handles localization for accessibility attributes (alt, title, aria-label)
 * and defaults the image source to `/logo.png`.
 *
 * @param props - Configuration for the logo, including source, styling, and interaction behavior.
 */
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
    const commonClassName = cx(
        'flex items-center gap-2 rounded-md px-2 py-1 border-0 hover:bg-slate-100 shrink-0',
        className
    )
    const resolvedTitle = title ?? t('components.logoButton.title')
    const resolvedAlt = alt ?? t('components.logoButton.alt')
    const ariaLabel = props['aria-label'] ?? t('components.logoButton.ariaLabel')

    const img = <img src={src} alt={resolvedAlt} className={cx('h-7 w-auto', imgClassName)} style={imgStyle} />

    if ('to' in props) {
        return (
            <Link to={props.to!} className={commonClassName} aria-label={ariaLabel} title={resolvedTitle}>
                {img}
            </Link>
        )
    }

    return (
        <Button
            type={props.type ?? 'button'}
            variant="unstyled"
            size="none"
            onClick={props.onClick}
            className={commonClassName}
            aria-label={ariaLabel}
            title={resolvedTitle}
        >
            {img}
        </Button>
    )
}
