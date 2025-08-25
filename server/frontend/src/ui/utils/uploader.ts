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

export async function uploadVideoFile(
    file: File,
    quality: UploadQuality,
    ctx: UploadContext,
    onProgress?: (percent: number) => void
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
    // TODO: const processed = await preprocessVideo(file)
    const processed = file

    // Step 3: Upload bytes and models to the job
    const form = new FormData()
    form.append('file', new Blob([await processed.arrayBuffer()], { type: processed.type }), processed.name)
    form.append('yolo_model', 'windsurfing/2025_08_09_100epochs.pt')
    form.append('reid_model', 'common/osnet_ain_x1_0_msmt17.pth')

    await doXhrUpload(`${API_BASE}/jobs/${job_id}/upload`, ctx.authHeader, form, onProgress)
    return 'uploaded'
}
