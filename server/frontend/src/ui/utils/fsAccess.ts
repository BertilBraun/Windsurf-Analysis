// Common filesystem helpers for File System Access API (Chrome/Edge)

export type FsEntry = {
    name: string
    relativePath: string
    kind: 'file' | 'directory'
    handle: FileSystemHandle

    getFile(): Promise<File>
}

function normalizeRelativePath(path: string): string {
    // Function to remove leading ./ and \ from the path
    return path.replace(/^[./\\]+/, '')
}

function getPathParts(path: string): string[] {
    // Function to split the path into parts (i.e. "a/b/c" -> ["a", "b", "c"])
    return path.split(/[\\/]+/).filter(Boolean)
}

export async function ensureReadPermission(
    dirHandle: FileSystemDirectoryHandle
): Promise<'granted' | 'denied' | 'prompt'> {
    const dh: any = dirHandle as any
    const qp = await dh.queryPermission?.({ mode: 'read' })
    if (qp === 'granted') return qp
    const rp = await dh.requestPermission?.({ mode: 'read' })
    return rp || 'denied'
}

export async function getFileByRelativePath(dirHandle: FileSystemDirectoryHandle, relativePath: string): Promise<File> {
    if ((await ensureReadPermission(dirHandle)) !== 'granted') throw new Error('Permission denied')

    const normalized = normalizeRelativePath(relativePath)
    const parts = getPathParts(normalized)
    try {
        let current: any = dirHandle
        for (let i = 0; i < parts.length; i++) {
            const name = parts[i]
            const isLast = i === parts.length - 1
            if (isLast) {
                current = await current.getFileHandle(name)
            } else {
                current = await current.getDirectoryHandle(name)
            }
        }
        const file = await (current as FileSystemFileHandle).getFile()
        return file
    } catch (e) {
        throw new Error(`Error getting file by relative path: ${e}`)
    }
}

async function* iterateEntries(
    dirHandle: FileSystemDirectoryHandle,
    basePath: string = ''
): AsyncGenerator<FsEntry, void, void> {
    const dh: any = dirHandle as any
    const hasEntries = typeof dh.entries === 'function'
    const hasValues = typeof dh.values === 'function'
    if (!hasEntries && !hasValues) return
    if (hasEntries) {
        for await (const [name, handle] of dh.entries()) {
            const rel = basePath ? `${basePath}${name}` : name
            yield {
                name,
                relativePath: rel,
                kind: handle.kind as any,
                handle,
                getFile: () => getFileFromHandle(handle),
            }
        }
    } else if (hasValues) {
        for await (const handle of dh.values()) {
            const name = (handle && (handle as any).name) || ''
            const rel = basePath ? `${basePath}${name}` : name
            yield {
                name,
                relativePath: rel,
                kind: handle.kind as any,
                handle,
                getFile: () => getFileFromHandle(handle),
            }
        }
    }
}

async function* iterateFilesRecursively(
    dirHandle: FileSystemDirectoryHandle,
    basePath: string = ''
): AsyncGenerator<FsEntry, void, void> {
    for await (const entry of iterateEntries(dirHandle, basePath)) {
        if (entry.kind === 'directory') {
            const subdir = entry.handle as unknown as FileSystemDirectoryHandle
            yield* iterateFilesRecursively(subdir, `${entry.relativePath}/`)
        } else if (entry.kind === 'file') {
            yield entry
        }
    }
}

export async function listFilesRecursively(
    dirHandle: FileSystemDirectoryHandle,
    extensions?: string[]
): Promise<FsEntry[]> {
    const out: FsEntry[] = []
    const normalizedExts = (extensions || []).map(e => e.toLowerCase())
    for await (const entry of iterateFilesRecursively(dirHandle, '')) {
        if (entry.kind !== 'file') continue
        if (normalizedExts.length > 0) {
            const lower = entry.name.toLowerCase()
            const matches = normalizedExts.some(ext => lower.endsWith(ext))
            if (!matches) continue
        }
        out.push(entry)
    }
    return out
}

async function getFileFromHandle(handle: FileSystemHandle): Promise<File> {
    if ((handle as any)?.kind !== 'file') throw new Error('Handle is not a file')
    const file = await (handle as FileSystemFileHandle).getFile()
    return file
}

export function isMp4Name(name: string): boolean {
    return /\.mp4$/i.test(name)
}

export function isMp4File(file: File): boolean {
    const type = (file.type || '').toLowerCase()
    return type === 'video/mp4'
}
