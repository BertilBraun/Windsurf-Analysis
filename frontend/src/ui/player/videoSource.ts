export type VideoSource =
    | { kind: 'ingress'; dirHandle: FileSystemDirectoryHandle | null }
    | { kind: 'file'; file: File }

