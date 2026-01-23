import shajs from 'sha.js'

const HASH_BLOCK_SIZE = 1024 * 1024 // 1 MiB
const SKIP_BLOCKS = 7 // hash 1 block, skip N blocks (~8x faster)
const YIELD_EVERY_SAMPLES = 8

export async function computeSha256(file: Blob): Promise<string> {
    const sha = shajs('sha256')
    const total = file.size || 0
    const blockSize = HASH_BLOCK_SIZE
    const step = blockSize * (SKIP_BLOCKS + 1)

    let sampled = 0
    for (let offset = 0; offset < total; offset += step) {
        const end = Math.min(total, offset + blockSize)
        const chunk = await file.slice(offset, end).arrayBuffer()
        sha.update(new Uint8Array(chunk))
        sampled += 1
        if (YIELD_EVERY_SAMPLES > 0 && sampled % YIELD_EVERY_SAMPLES === 0) {
            await new Promise(resolve => window.setTimeout(resolve, 0))
        }
    }

    const lastStart = Math.max(0, total - blockSize)
    const lastChunk = await file.slice(lastStart, total).arrayBuffer()
    sha.update(new Uint8Array(lastChunk))
    return String(sha.digest('hex')).toLowerCase()
}

export type FileFingerprint = {
    path: string
    size: number
    mtimeMs: number
    sha256: string
}

export type FileSnapshot = {
    fileFingerprints: FileFingerprint[]
    updatedAt: number
}

export function normalizeRelativePath(path: string): string {
    return String(path || '')
        .replace(/^[./\\]+/, '')
        .replace(/\\/g, '/')
}

export function buildShaToPaths(snapshot: FileSnapshot | null): Map<string, string[]> {
    const shaToPaths = new Map<string, string[]>()
    if (!snapshot) return shaToPaths

    for (const fingerprint of snapshot.fileFingerprints) {
        const sha = fingerprint.sha256
        const path = normalizeRelativePath(fingerprint.path)
        const existingPaths = shaToPaths.get(sha)
        if (existingPaths) existingPaths.push(path)
        else shaToPaths.set(sha, [path])
    }

    for (const [sha, paths] of shaToPaths.entries()) {
        const unique = Array.from(new Set(paths.filter(Boolean)))
        unique.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()))
        shaToPaths.set(sha, unique)
    }

    return shaToPaths
}
