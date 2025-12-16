/** @type {import('tailwindcss').Config} */
export default {
    content: ['./index.html', './src/**/*.{js,ts,jsx,tsx,html}'],
    theme: {
        extend: {
            colors: {
                brand: {
                    50: 'var(--brand-50)',
                    600: 'var(--brand-600)',
                    700: 'var(--brand-700)',
                },
            },
        },
    },
    plugins: [],
}


