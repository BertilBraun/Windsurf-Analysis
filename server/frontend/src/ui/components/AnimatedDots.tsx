import React from 'react'

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
