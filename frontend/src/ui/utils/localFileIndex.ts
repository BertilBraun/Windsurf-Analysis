/**
 * @fileoverview Utilities for indexing and fingerprinting local files.
 */

import shajs from 'sha.js'

const HASH_BLOCK_SIZE = 1024 * 1024 // 1 MiB
const SKIP_BLOCKS = 7 // hash 1 block, skip N blocks (~8x faster)
const YIELD_EVERY_SAMPLES = 8

/**
 * Computes a sampled SHA-256 hash of a file for fingerprinting.
 * Uses a sampling strategy to improve performance on large files by skipping blocks
 * and periodically yielding to the main thread.
 *
 * @param file - The file blob to hash.
 * @returns A promise that resolves to the hex-encoded SHA-256 hash.
 */
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

/**
 * Metadata representing a unique file state.
 */
export type FileFingerprint = {
    /** Relative path of the file. */
    path: string
    /** File size in bytes. */
    size: number
    /** Last modification time in milliseconds. */
    mtimeMs: number
    /** Sampled SHA-256 hash of the file content. */
    sha256: string
}

/**
 * A collection of file fingerprints captured at a specific point in time.
 */
export type FileSnapshot = {
    /** Array of fingerprints for files included in the snapshot. */
    fileFingerprints: FileFingerprint[]
    /** Timestamp (ms) when the snapshot was generated. */
    updatedAt: number
}

/**
 * Normalizes a relative file path by removing leading separators and standardizing slashes.
 *
 * @param path - The raw file path to normalize.
 * @returns The normalized path string.
 */
export function normalizeRelativePath(path: string): string {
    return String(path || '')
        .replace(/^[./\\]+/, '')
        .replace(/\\/g, '/')
}

/**
 * Builds a map of SHA-256 hashes to their corresponding normalized file paths from a snapshot.
 *
 * @param snapshot - The file snapshot to process, or null.
 * @returns A map where keys are SHA-256 hashes and values are arrays of unique normalized paths.
 */
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
