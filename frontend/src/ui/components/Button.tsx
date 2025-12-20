import React from 'react'
import { useTranslation } from 'react-i18next'
import { AnimatedDots } from './AnimatedDots'

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
    text: string
    isPending?: boolean
    variant?: 'primary' | 'secondary' | 'danger' | 'ghost'
    size?: 'sm' | 'md'
}

function cx(...parts: Array<string | undefined | null | false>) {
    return parts.filter(Boolean).join(' ')
}

export const Button: React.FC<ButtonProps> = ({
    text,
    isPending,
    disabled,
    variant = 'secondary',
    size = 'md',
    className,
    ...props
}) => {
    const { t } = useTranslation()
    const variantClass =
        variant === 'primary'
            ? 'bg-brand-600 text-white hover:bg-brand-700'
            : variant === 'danger'
              ? 'bg-red-600 text-white hover:bg-red-700'
              : variant === 'ghost'
                ? 'bg-transparent text-slate-700 hover:bg-slate-100'
                : 'bg-slate-900 text-white hover:bg-slate-800'

    const sizeClass = size === 'sm' ? 'px-2.5 py-1.5 text-xs' : 'px-3 py-2 text-sm'

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
            {isPending ? (
                <span>
                    {text}
                    {t('components.button.pendingSuffix')} <AnimatedDots />
                </span>
            ) : (
                text
            )}
        </button>
    )
}
