import React from 'react'
import { AnimatedDots } from './AnimatedDots'

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
    text: string
    isPending?: boolean
}

export const Button: React.FC<ButtonProps> = ({ text, isPending, ...props }) => {
    return (
        <button {...props} disabled={isPending}>
            {isPending ? (
                <span>
                    {text}ing <AnimatedDots />
                </span>
            ) : (
                text
            )}
        </button>
    )
}
