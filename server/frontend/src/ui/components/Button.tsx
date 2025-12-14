import React from 'react'
import { AnimatedDots } from './AnimatedDots'

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
    text: string
    isPending?: boolean
}

export const Button: React.FC<ButtonProps> = ({ text, isPending, disabled, ...props }) => {
    return (
        <button
            {...props}
            disabled={!!disabled || !!isPending}
            className="px-2 py-1 rounded-md bg-gray-700 text-gray-100 hover:bg-gray-600 hover:cursor-pointer"
        >
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
