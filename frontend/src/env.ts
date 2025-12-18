export function requireEnv(name: string): string {
    const value = (import.meta.env as Record<string, string | undefined>)[name]
    if (!value || value.trim() === '' || value === 'REPLACE_ME') {
        throw new Error(
            `Missing or invalid ${name}. Create frontend/.env.local using frontend/env.example and restart \`npm run dev\`.`
        )
    }
    return value
}
