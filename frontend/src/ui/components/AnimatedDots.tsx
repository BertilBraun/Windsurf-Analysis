/**
 * @module AnimatedDots
 * @description Provides a simple loading indicator component.
 */

import React from 'react'

/**
 * Renders an animated sequence of dots (., .., ...) that cycles every 500ms.
 * Maintains a fixed width to prevent layout shifts during the animation.
 *
 * @returns {JSX.Element} A span element containing the cycling dots.
 */
export const AnimatedDots: React.FC = () => {
    const [dots, setDots] = React.useState('.')

    React.useEffect(() => {
        const interval = setInterval(() => {
            setDots(prev => (prev === '...' ? '.' : prev + '.'))
        }, 500)
        return () => clearInterval(interval)
    }, [])

    // make sure that it is always the same width regardless of the number of dots
    return <span style={{ width: '100px' }}>{dots}</span>
}
