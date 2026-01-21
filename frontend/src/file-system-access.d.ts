type FileSystemPermissionMode = 'read' | 'readwrite'

type FileSystemHandlePermissionDescriptor = {
    mode?: FileSystemPermissionMode
}

interface FileSystemDirectoryHandle {
    queryPermission(descriptor?: FileSystemHandlePermissionDescriptor): Promise<PermissionState>
    requestPermission(descriptor?: FileSystemHandlePermissionDescriptor): Promise<PermissionState>
}

interface Window {
    showDirectoryPicker(options?: { id?: string }): Promise<FileSystemDirectoryHandle>
}

