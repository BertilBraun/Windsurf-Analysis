export function getConcurrency(): number {
    return Math.max(1, Math.floor(navigator.hardwareConcurrency ?? 4))
}

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
