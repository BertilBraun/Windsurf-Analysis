import React from 'react'
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
    title = 'GybeLock',
    src = '/logo.png',
    alt = 'GybeLock',
    ...props
}) => {
    const commonClassName = cx('flex items-center gap-2 rounded-md px-2 py-1 hover:bg-slate-100', className)
    const ariaLabel = props['aria-label'] ?? 'GybeLock Home'

    const img = <img src={src} alt={alt} className={cx('h-7 w-auto', imgClassName)} style={imgStyle} />

    if ('to' in props) {
        return (
            <Link to={props.to} className={commonClassName} aria-label={ariaLabel} title={title}>
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
            title={title}
        >
            {img}
        </button>
    )
}


