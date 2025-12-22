export async function computeSha256(file: File): Promise<{ arrayBuffer: ArrayBuffer; sha256: string }> {
    const arrayBuffer = await file.arrayBuffer()
    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer)
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    const sha256 = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
    const sha256Lower = String(sha256).toLowerCase()
    return { arrayBuffer, sha256: sha256Lower }
}

export type FileFingerprint = {
    path: string
    size: number
    mtimeMs: number
}

export type FileSnapshot = {
    files: FileFingerprint[]
    fingerprintToSha: Record<string, string>
    updatedAt: number
}

export function normalizeRelativePath(path: string): string {
    return String(path || '')
        .replace(/^[./\\]+/, '')
        .replace(/\\/g, '/')
}

export function fingerprintKey(fp: FileFingerprint): string {
    const path = normalizeRelativePath(fp.path)
    return `${path}|${fp.size}|${fp.mtimeMs}`
}

export function getFingerprintSha(snapshot: FileSnapshot, fp: FileFingerprint): string | null {
    const key = fingerprintKey(fp)
    return snapshot.fingerprintToSha[key] ?? null
}

export function getNewFingerprints(prev: FileSnapshot | null, next: FileSnapshot): FileFingerprint[] {
    const prevKeys = new Set<string>()
    if (prev) {
        for (const fp of prev.files) prevKeys.add(fingerprintKey(fp))
    }
    return next.files.filter(fp => !prevKeys.has(fingerprintKey(fp)))
}

export function buildShaToPaths(snapshot: FileSnapshot | null): Map<string, string[]> {
    const map = new Map<string, string[]>()
    if (!snapshot) return map

    for (const fp of snapshot.files) {
        const sha = getFingerprintSha(snapshot, fp)
        if (!sha) continue
        const path = normalizeRelativePath(fp.path)
        const existing = map.get(sha)
        if (existing) existing.push(path)
        else map.set(sha, [path])
    }

    for (const [sha, paths] of map.entries()) {
        const uniq = Array.from(new Set(paths.filter(Boolean)))
        uniq.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()))
        map.set(sha, uniq)
    }

    return map
}
