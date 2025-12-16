import { auth, modalUrl } from '../../firebase'

const CHUNK_SIZE = 8 * 1024 * 1024 // 8 MiB

export async function computeSha256(file: File): Promise<string> {
    const buf = await file.arrayBuffer()
    const hashBuffer = await crypto.subtle.digest('SHA-256', buf)
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
}

async function getBearer(): Promise<string> {
    const user = auth.currentUser
    if (!user) throw new Error('Not signed in.')
    const token = await user.getIdToken()
    return `Bearer ${token}`
}

async function xhrUploadPart(url: string, bearer: string, form: FormData, onProgress?: (p: number) => void) {
    await new Promise<void>((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open('POST', url, true)
        xhr.setRequestHeader('Authorization', bearer)
        xhr.upload.onprogress = e => {
            if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total)
        }
        xhr.onerror = () => reject(new Error('Network error'))
        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) resolve()
            else reject(new Error(xhr.responseText || `HTTP ${xhr.status}`))
        }
        xhr.send(form)
    })
}

export async function uploadToModal(jobId: string, file: File, onProgress?: (p: number) => void) {
    const bearer = await getBearer()
    const processed = await file.arrayBuffer()
    const totalSize = processed.byteLength
    const totalParts = Math.ceil(totalSize / CHUNK_SIZE)

    const initForm = new FormData()
    initForm.append('total_size', String(totalSize))
    initForm.append('chunk_size', String(CHUNK_SIZE))
    initForm.append('total_parts', String(totalParts))
    initForm.append('file_name', file.name)
    initForm.append('mime_type', file.type || 'video/mp4')
    initForm.append('yolo_model', 'windsurfing/best.pt')

    const initRes = await fetch(`${modalUrl}/jobs/${jobId}/upload/init`, {
        method: 'POST',
        headers: { Authorization: bearer },
        body: initForm,
    })
    if (!initRes.ok) throw new Error(await initRes.text())
    const { resume_from_part } = (await initRes.json()) as { resume_from_part: number }

    const view = new Uint8Array(processed)
    let uploadedBytes = 0
    for (let i = 0; i < resume_from_part; i++) {
        const start = i * CHUNK_SIZE
        const end = Math.min(start + CHUNK_SIZE, totalSize)
        uploadedBytes += end - start
    }

    for (let partIndex = resume_from_part; partIndex < totalParts; partIndex++) {
        const start = partIndex * CHUNK_SIZE
        const end = Math.min(start + CHUNK_SIZE, totalSize)
        const chunkBytes = view.subarray(start, end)
        const partSize = end - start

        const form = new FormData()
        form.append('part_index', String(partIndex))
        form.append(
            'chunk',
            new Blob([chunkBytes], { type: file.type || 'application/octet-stream' }),
            `${file.name}.part${partIndex}`
        )

        await xhrUploadPart(`${modalUrl}/jobs/${jobId}/upload/part`, bearer, form, p => {
            if (!onProgress) return
            // best-effort per-part progress
            const current = uploadedBytes + Math.round(p * partSize)
            onProgress(Math.min(0.99, current / totalSize))
        })

        uploadedBytes += partSize
        if (onProgress) onProgress(Math.min(0.99, uploadedBytes / totalSize))
    }

    const completeRes = await fetch(`${modalUrl}/jobs/${jobId}/upload/complete`, {
        method: 'POST',
        headers: { Authorization: bearer },
    })
    if (!completeRes.ok) throw new Error(await completeRes.text())
    if (onProgress) onProgress(1)
}

