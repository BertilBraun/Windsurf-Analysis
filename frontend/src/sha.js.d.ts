/**
 * Type definitions for the sha.js library.
 */
declare module 'sha.js' {
    /**
     * Supported encoding for the digest output.
     */
    type DigestEncoding = 'hex'

    /**
     * Represents a hash object for computing cryptographic digests.
     */
    type Hash = {
        /**
         * Updates the hash content with the given data.
         * @param data The data to be hashed.
         * @returns The hash instance for chaining.
         */
        update(data: string | ArrayBuffer | ArrayBufferView): Hash
        /**
         * Calculates the digest of all data passed to the hash.
         * @param encoding The encoding to use for the returned string.
         * @returns The calculated digest.
         */
        digest(encoding: DigestEncoding): string
    }

    /**
     * Creates a hash instance for the specified algorithm.
     * @param algorithm The hashing algorithm to use.
     * @returns A new Hash instance.
     */
    export default function shajs(algorithm: 'sha256'): Hash
}
