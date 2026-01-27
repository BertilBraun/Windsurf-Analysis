/**
 * Environment variable utilities for the frontend.
 */

/**
 * Retrieves a required environment variable from the application environment.
 *
 * @param name - The name of the environment variable to retrieve.
 * @returns The value of the environment variable.
 * @throws Error if the variable is missing, empty, or set to 'REPLACE_ME'.
 */
export function requireEnv(name: string): string {
    const value = (import.meta.env as Record<string, string | undefined>)[name]
    if (!value || value.trim() === '' || value === 'REPLACE_ME') {
        throw new Error(
            `Missing or invalid ${name}. Create frontend/.env.local using frontend/env.example and restart \`npm run dev\`.`
        )
    }
    return value
}
