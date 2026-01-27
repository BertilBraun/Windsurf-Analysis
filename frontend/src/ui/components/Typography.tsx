/**
 * @module Typography
 * Provides a set of components for consistent text styling, hierarchy, and layout.
 */
import React from 'react'

function cx(...parts: Array<string | undefined | null | false>) {
    return parts.filter(Boolean).join(' ')
}

/**
 * Configuration properties for the {@link Heading} component.
 */
export type HeadingProps = Omit<React.HTMLAttributes<HTMLHeadingElement>, 'color'> & {
    /** The heading level (1-3), determining the HTML tag and base font size. */
    level: 1 | 2 | 3
    /** The color tone of the heading. Defaults to 'default'. */
    tone?: 'default' | 'brand'
}

/**
 * Renders a semantic heading element (h1, h2, or h3) with predefined typography styles.
 */
export const Heading: React.FC<HeadingProps> = ({ level, tone = 'default', className, ...props }) => {
    const Tag = level === 1 ? 'h1' : level === 2 ? 'h2' : 'h3'
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

/**
 * Configuration properties for the {@link Text} component.
 */
export type TextProps = Omit<React.HTMLAttributes<HTMLElement>, 'color'> & {
    /** The HTML element to render. Defaults to 'div'. */
    as?: 'div' | 'p' | 'span'
    /** The visual variant of the text, affecting size and line height. Defaults to 'body'. */
    variant?: 'body' | 'support' | 'muted'
    /** The font weight. Defaults to 'normal'. */
    weight?: 'normal' | 'medium' | 'semibold' | 'bold'
    /** The color tone. Defaults to 'default'. */
    tone?: 'default' | 'brand' | 'danger'
}

/**
 * Renders a text element with configurable size, weight, and color.
 */
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

/**
 * Configuration properties for the {@link TextStack} component.
 */
export type TextStackProps = React.HTMLAttributes<HTMLDivElement> & {
    /** The visual variant of the text within the stack. Defaults to 'body'. */
    variant?: 'body' | 'support' | 'muted'
}

/**
 * A container component that applies vertical spacing and consistent text styling to its children.
 */
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

/**
 * Renders a `<strong>` element with standard bold styling and high-contrast slate color.
 */
export const TextStrong: React.FC<React.HTMLAttributes<HTMLElement>> = ({ className, ...props }) => {
    return <strong {...props} className={cx('font-semibold text-slate-900', className)} />
}

/**
 * Renders a `<strong>` element with brand-colored bold styling.
 */
export const BrandStrong: React.FC<React.HTMLAttributes<HTMLElement>> = ({ className, ...props }) => {
    return <strong {...props} className={cx('font-semibold text-brand-700', className)} />
}
