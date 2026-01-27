/**
 * Type definitions for the File System Access API.
 * Provides interfaces for interacting with the local file system.
 */

/**
 * Permission mode for file system handles, specifying read or read-write access.
 */
type FileSystemPermissionMode = 'read' | 'readwrite'

/**
 * Options for querying or requesting permissions on a file system handle.
 */
type FileSystemHandlePermissionDescriptor = {
    /**
     * The permission mode to check or request.
     */
    mode?: FileSystemPermissionMode
}

/**
 * Represents a handle to a file system directory.
 */
interface FileSystemDirectoryHandle {
    /**
     * Queries the current permission state of the handle.
     * @param descriptor The permission descriptor.
     * @returns A promise that resolves to the current permission state ('granted', 'denied', or 'prompt').
     */
    queryPermission(descriptor?: FileSystemHandlePermissionDescriptor): Promise<PermissionState>

    /**
     * Requests permission for the handle.
     * @param descriptor The permission descriptor.
     * @returns A promise that resolves to the resulting permission state.
     */
    requestPermission(descriptor?: FileSystemHandlePermissionDescriptor): Promise<PermissionState>
}

interface Window {
    /**
     * Displays a directory picker to allow the user to select a directory.
     * @param options Configuration options for the directory picker.
     * @param options.id An optional unique ID to remember the last opened directory for this specific picker.
     * @returns A promise that resolves to a handle for the selected directory.
     */
    showDirectoryPicker(options?: { id?: string }): Promise<FileSystemDirectoryHandle>
}
