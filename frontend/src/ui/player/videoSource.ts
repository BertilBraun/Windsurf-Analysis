/**
 * @fileoverview Defines the types of video sources supported by the player,
 * including local file system handles and individual file objects.
 */

/**
 * Represents the origin of a video stream or file for the player.
 */
export type VideoSource =
    /**
     * A source derived from a local directory handle, typically used for processing
     * multiple files or streaming from a local workspace.
     */
    | { kind: 'ingress'; dirHandle: FileSystemDirectoryHandle | null }
    /**
     * A source derived from a single specific file object.
     */
    | { kind: 'file'; file: File }
