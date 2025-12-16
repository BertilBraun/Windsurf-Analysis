export const assert = (condition: boolean, message: string | null = null) => {
    if (!condition) throw new Error(message || 'Assertion failed')
}
