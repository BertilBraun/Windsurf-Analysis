import { API_BASE } from '../auth/AuthProvider'
import { preprocessVideo } from './preprocessVideo'
import { UploadQuality } from '../types'

export type UploadContext = {
    authorizedFetch: (input: RequestInfo, init?: RequestInit) => Promise<Response>
    authHeader: string | null
}

export async function computeSha256(file: File): Promise<{ arrayBuffer: ArrayBuffer; sha256: string }> {
    const arrayBuffer = await file.arrayBuffer()
    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer)
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    const sha256 = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
    return { arrayBuffer, sha256 }
}

export function doXhrUpload(
    url: string,
    authHeader: string | null,
    form: FormData,
    onProgress?: (percent: number) => void
): Promise<void> {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open('POST', url, true)
        if (authHeader) xhr.setRequestHeader('Authorization', authHeader)
        xhr.upload.onprogress = e => {
            if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100))
        }
        xhr.onerror = () => reject(new Error('Network error'))
        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) resolve()
            else reject(new Error(xhr.responseText || `HTTP ${xhr.status}`))
        }
        xhr.send(form)
    })
}

const CHUNK_SIZE = 8 * 1024 * 1024 // 8 MiB per part

export async function uploadVideoFile(
    file: File,
    quality: UploadQuality,
    ctx: UploadContext,
    onProgress: (percent: number) => void
): Promise<'uploaded' | 'skipped'> {
    // Step 1: Create job (also acts as duplicate/quota check)
    const { sha256 } = await computeSha256(file)
    const createRes = await ctx.authorizedFetch('/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ original_checksum_sha256: sha256 }),
    })
    if (createRes.status === 409) return 'skipped'
    if (createRes.status === 403) {
        let err: any = new Error('Quota exceeded')
        err.code = 'quota_exceeded'
        err.status = 403
        throw err
    }
    if (!createRes.ok) throw new Error(await createRes.text())
    const { job_id } = (await createRes.json()) as { job_id: string; status: string }

    // Step 2: Preprocess
    const processed = await preprocessVideo(file, quality)
    const processedBuffer = processed instanceof ArrayBuffer ? processed : (processed as Uint8Array).buffer
    const totalSize = processedBuffer.byteLength
    const totalParts = Math.ceil(totalSize / CHUNK_SIZE)

    // Step 3: INIT chunked upload (also carries model params)
    const initForm = new FormData()
    initForm.append('total_size', String(totalSize))
    initForm.append('chunk_size', String(CHUNK_SIZE))
    initForm.append('total_parts', String(totalParts))
    initForm.append('file_name', file.name)
    initForm.append('mime_type', file.type || 'video/mp4')
    initForm.append('yolo_model', 'windsurfing/2025_08_09_100epochs.pt')
    initForm.append('reid_model', 'common/osnet_ain_x1_0_msmt17.pth')

    const initRes = await ctx.authorizedFetch(`/jobs/${job_id}/upload/init`, {
        method: 'POST',
        body: initForm,
    })
    if (!initRes.ok) throw new Error(await initRes.text())
    const { resume_from_part } = (await initRes.json()) as { resume_from_part: number }

    // Step 4: Upload parts
    const processedView = new Uint8Array(processedBuffer)
    let uploadedBytesBeforeCurrentPart = resume_from_part * CHUNK_SIZE
    for (let partIndex = resume_from_part; partIndex < totalParts; partIndex++) {
        const start = partIndex * CHUNK_SIZE
        const end = Math.min(start + CHUNK_SIZE, totalSize)
        const chunkBytes = processedView.subarray(start, end)
        const partSize = end - start

        const partForm = new FormData()
        partForm.append('part_index', String(partIndex))
        partForm.append(
            'chunk',
            new Blob([chunkBytes], { type: file.type || 'application/octet-stream' }),
            `${file.name}.part${partIndex}`
        )

        await doXhrUpload(`${API_BASE}/jobs/${job_id}/upload/part`, ctx.authHeader, partForm, (percent: number) => {
            const partUploaded = Math.round((percent / 100) * partSize)
            const overall = Math.floor(((uploadedBytesBeforeCurrentPart + partUploaded) / totalSize) * 100)
            onProgress(Math.min(99, overall))
        })

        uploadedBytesBeforeCurrentPart += partSize
        onProgress(Math.min(99, Math.floor((uploadedBytesBeforeCurrentPart / totalSize) * 100)))
    }

    // Step 5: COMPLETE upload (server concatenates, checksums, and spawns inference)
    const completeRes = await ctx.authorizedFetch(`/jobs/${job_id}/upload/complete`, {
        method: 'POST',
    })
    if (!completeRes.ok) throw new Error(await completeRes.text())
    onProgress(100)
    return 'uploaded'
}
