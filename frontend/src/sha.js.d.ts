declare module 'sha.js' {
    type DigestEncoding = 'hex'

    type Hash = {
        update(data: string | ArrayBuffer | ArrayBufferView): Hash
        digest(encoding: DigestEncoding): string
    }

    export default function shajs(algorithm: 'sha256'): Hash
}

