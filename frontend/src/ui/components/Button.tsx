import React from 'react'
import { useTranslation } from 'react-i18next'
import { Spinner } from './Spinner'

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
    text?: string
    children?: React.ReactNode
    isPending?: boolean
    variant?: 'primary' | 'secondary' | 'danger' | 'ghost' | 'outline' | 'brandOutline' | 'warning' | 'inverse' | 'unstyled'
    size?: 'sm' | 'md' | 'none'
}

function cx(...parts: Array<string | undefined | null | false>) {
    return parts.filter(Boolean).join(' ')
}

export const Button: React.FC<ButtonProps> = ({
    text,
    children,
    isPending,
    disabled,
    variant = 'secondary',
    size = 'sm',
    className,
    ...props
}) => {
    const { t } = useTranslation()
    const variantClass =
        variant === 'primary'
            ? 'bg-brand-600 text-white hover:bg-brand-700'
            : variant === 'brandOutline'
            ? 'border border-brand-600/40 bg-white text-brand-700 hover:bg-brand-50'
            : variant === 'outline'
            ? 'border border-slate-200 bg-white text-slate-800 hover:bg-slate-50'
            : variant === 'warning'
            ? 'bg-amber-500 text-white hover:bg-amber-600'
            : variant === 'inverse'
            ? 'bg-white/10 text-gray-100 hover:bg-white/20'
            : variant === 'danger'
            ? 'bg-red-600 text-white hover:bg-red-700'
            : variant === 'ghost'
            ? 'bg-transparent text-slate-700 hover:bg-slate-100'
            : variant === 'unstyled'
            ? ''
            : 'bg-slate-900 text-white hover:bg-slate-800'

    const sizeClass = size === 'none' ? '' : size === 'sm' ? 'px-2.5 py-1.5 text-xs' : 'px-3 py-2 text-sm'

    const content =
        isPending && text ? (
            <span>
                {text}&nbsp;
                <Spinner />
            </span>
        ) : (
            children ?? text
        )

    return (
        <button
            {...props}
            disabled={!!disabled || !!isPending}
            className={cx(
                'inline-flex items-center justify-center gap-2 rounded-md font-medium transition active:translate-y-[0.5px] cursor-pointer enabled:hover:scale-[1.02] disabled:opacity-60 disabled:cursor-not-allowed',
                sizeClass,
                variantClass,
                className
            )}
        >
            {content}
        </button>
    )
}
