import shajs from 'sha.js'
import { assert } from './assert'

const HASH_CHUNK_SIZE = 8 * 1024 * 1024 // 8 MiB

export async function computeSha256(file: Blob): Promise<{ sha256: string }> {
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
    return { sha256: String(sha.digest('hex')).toLowerCase() }
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

export function getFingerprintSha(snapshot: FileSnapshot, fp: FileFingerprint): string {
    const key = fingerprintKey(fp)
    assert(snapshot.fingerprintToSha[key] !== undefined, 'Fingerprint not found in snapshot')
    return snapshot.fingerprintToSha[key]
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
