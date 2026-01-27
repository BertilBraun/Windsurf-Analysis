/**
 * Utilities for managing and limiting concurrent asynchronous operations.
 */

/**
 * Returns the recommended number of concurrent operations based on hardware capabilities.
 * Defaults to 4 if hardware concurrency information is unavailable.
 *
 * @returns A positive integer representing the concurrency level.
 */
export function getConcurrency(): number {
    return Math.max(1, Math.floor(navigator.hardwareConcurrency ?? 4))
}

/**
 * Processes an array of items using an asynchronous function with a concurrency limit.
 *
 * @param items - The array of items to process.
 * @param fn - The asynchronous function to apply to each item.
 * @param limit - The maximum number of concurrent operations. Defaults to the value from {@link getConcurrency}.
 * @returns A promise that resolves when all items have been processed.
 */
export async function mapLimit<T>(items: T[], fn: (item: T) => Promise<void>, limit?: number): Promise<void> {
    let nextIdx = 0
    async function worker(): Promise<void> {
        while (true) {
            const i = nextIdx
            if (i >= items.length) return
            nextIdx = i + 1
            await fn(items[i])
        }
    }
    const n = Math.max(1, Math.min(limit || getConcurrency(), items.length))
    await Promise.all(Array.from({ length: n }, () => worker()))
}
