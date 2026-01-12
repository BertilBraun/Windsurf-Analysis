import shajs from 'sha.js'

const HASH_CHUNK_SIZE = 8 * 1024 * 1024 // 8 MiB

export async function computeSha256(file: Blob): Promise<string> {
    const sha = shajs('sha256')
    const total = file.size || 0
    let offset = 0
    while (offset < total) {
        const end = Math.min(total, offset + HASH_CHUNK_SIZE)
        const chunk = await file.slice(offset, end).arrayBuffer()
        sha.update(new Uint8Array(chunk))
        offset = end
        if (offset % (HASH_CHUNK_SIZE * 2) === 0) await new Promise(resolve => window.setTimeout(resolve, 0))
    }
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
