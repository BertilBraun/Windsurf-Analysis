/**
 * @fileoverview Spinner component for displaying loading states.
 */

/**
 * A circular loading indicator with customizable size.
 *
 * @param props - The component props.
 * @param props.size - The size variant of the spinner. Defaults to 'small'.
 */
export const Spinner: React.FC<{ size?: 'small' | 'medium' | 'large' }> = ({ size = 'small' }) => {
    const sizeClass = size === 'small' ? 'w-3 h-3' : size === 'medium' ? 'w-8 h-8' : 'w-12 h-12'
    return (
        <span
            className={`inline-block ${sizeClass} rounded-full border-2 border-brand-600/30 border-t-brand-600 animate-spin`}
        />
    )
}
