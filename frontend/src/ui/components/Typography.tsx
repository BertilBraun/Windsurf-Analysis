import React from 'react'

function cx(...parts: Array<string | undefined | null | false>) {
    return parts.filter(Boolean).join(' ')
}

export type HeadingProps = Omit<React.HTMLAttributes<HTMLHeadingElement>, 'color'> & {
    level: 1 | 2 | 3
    tone?: 'default' | 'brand'
}

export const Heading: React.FC<HeadingProps> = ({ level, tone = 'default', className, ...props }) => {
    const Tag = (level === 1 ? 'h1' : level === 2 ? 'h2' : 'h3') as const
    const baseClass =
        level === 1
            ? 'text-3xl sm:text-4xl font-semibold tracking-tight'
            : level === 2
            ? 'text-xl sm:text-2xl font-semibold'
            : 'text-lg font-semibold'

    const spacingClass = level === 1 ? 'mt-0 mb-2' : 'mt-0 mb-1.5'
    const toneClass = tone === 'brand' ? 'text-brand-700' : 'text-slate-900'

    return <Tag {...props} className={cx(spacingClass, baseClass, toneClass, className)} />
}

export type TextProps = Omit<React.HTMLAttributes<HTMLElement>, 'color'> & {
    as?: 'div' | 'p' | 'span'
    variant?: 'body' | 'support' | 'muted'
    weight?: 'normal' | 'medium' | 'semibold' | 'bold'
    tone?: 'default' | 'brand' | 'danger'
}

export const Text: React.FC<TextProps> = ({
    as: Tag = 'div',
    variant = 'body',
    weight = 'normal',
    tone = 'default',
    className,
    ...props
}) => {
    const sizeClass =
        variant === 'muted'
            ? 'text-xs leading-5'
            : variant === 'support'
            ? 'text-sm leading-6'
            : 'text-sm sm:text-base leading-6'

    const colorClass =
        tone === 'danger'
            ? 'text-red-600'
            : tone === 'brand'
            ? 'text-brand-700'
            : variant === 'muted'
            ? 'text-slate-500'
            : 'text-slate-700'

    const weightClass =
        weight === 'bold'
            ? 'font-bold'
            : weight === 'semibold'
            ? 'font-semibold'
            : weight === 'medium'
            ? 'font-medium'
            : ''

    return <Tag {...props} className={cx(sizeClass, colorClass, weightClass, className)} />
}

export type TextStackProps = React.HTMLAttributes<HTMLDivElement> & {
    variant?: 'body' | 'support' | 'muted'
}

export const TextStack: React.FC<TextStackProps> = ({ variant = 'body', className, ...props }) => {
    const sizeClass =
        variant === 'muted'
            ? 'text-xs leading-5'
            : variant === 'support'
            ? 'text-sm leading-6'
            : 'text-sm sm:text-base leading-6'

    const colorClass = variant === 'muted' ? 'text-slate-500' : 'text-slate-700'

    return <div {...props} className={cx(sizeClass, colorClass, 'space-y-3', className)} />
}

export const TextStrong: React.FC<React.HTMLAttributes<HTMLElement>> = ({ className, ...props }) => {
    return <strong {...props} className={cx('font-semibold text-slate-900', className)} />
}

export const BrandStrong: React.FC<React.HTMLAttributes<HTMLElement>> = ({ className, ...props }) => {
    return <strong {...props} className={cx('font-semibold text-brand-700', className)} />
}
