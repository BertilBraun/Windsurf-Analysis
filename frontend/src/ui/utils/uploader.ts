import { modalUrl } from '../../firebase'
import { getVideoTrack } from '../hooks/useVideoFps'
import { UploadQuality } from '../types'
import { trackEvent } from './analytics'
import { computeSha256 } from './localFileIndex'

export type UploadContext = {
    authorizedFetch: (input: RequestInfo, init?: RequestInit) => Promise<Response>
    getAuthHeader: () => Promise<string>
}

const MODAL_BASE = modalUrl + '/api/v1'

const MAX_PARALLEL_UPLOAD_REQUESTS = 16
const MAX_PARALLEL_VIDEO_UPLOADS = 3

function createLimiter(max: number) {
    const waiters: Array<() => void> = []
    let inFlight = 0

    return async (): Promise<() => void> => {
        let released = false
        const release = () => {
            if (released) return
            released = true
            if (inFlight > 0) inFlight -= 1
            const next = waiters.shift()
            if (next) next()
        }

        if (inFlight < max) {
            inFlight += 1
            return release
        }

        return new Promise(resolve => {
            waiters.push(() => {
                inFlight += 1
                resolve(release)
            })
        })
    }
}

const acquireUploadSlot = createLimiter(MAX_PARALLEL_UPLOAD_REQUESTS)
const acquireVideoUploadSlot = createLimiter(MAX_PARALLEL_VIDEO_UPLOADS)

export async function doXhrUpload(
    url: string,
    getAuthHeader: (() => Promise<string>) | null,
    form: FormData,
    onProgress?: (percent: number) => void
): Promise<void> {
    const release = await acquireUploadSlot()
    try {
        const authHeader = getAuthHeader ? await getAuthHeader() : null
        await new Promise<void>((resolve, reject) => {
            const xhr = new XMLHttpRequest()
            xhr.open('POST', url, true)
            if (authHeader) xhr.setRequestHeader('Authorization', authHeader)
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
    } finally {
        release()
    }
}

const CHUNK_SIZE = 8 * 1024 * 1024 // 8 MiB per part
const MAX_CONCURRENCY = Math.min(
    MAX_PARALLEL_UPLOAD_REQUESTS,
    Math.max(
        2,
        typeof navigator !== 'undefined' && (navigator as any).hardwareConcurrency
            ? Math.floor((navigator as any).hardwareConcurrency / 2)
            : 4
    )
)

// TODO with preprocess const PERCENT_PREPROCESS = 0.3
// TODO with preprocess const PERCENT_UPLOAD = 0.7
const PERCENT_PREPROCESS = 0.0
const PERCENT_UPLOAD = 0.95

export async function uploadVideoFile(params: {
    file: File
    quality: UploadQuality
    ctx: UploadContext
    onProgress: (percent: number) => void
    onStarted?: () => void
    precomputedSha256?: string
}): Promise<'uploaded' | 'skipped'> {
    const { file, quality, ctx, onProgress, onStarted, precomputedSha256 } = params

    // Step 1: Create job (also acts as duplicate/quota check)
    trackEvent('analysis_upload_start', {
        file_size_bytes: file.size,
        mime: file.type || 'video/mp4',
        quality,
    })

    // get number of frames of the video and skip if longer than MAX_FRAMES
    const video = await getVideoTrack(file)
    const frameCount = video?.FrameCount
    const MAX_FRAMES = 30 * 60 * 3 // 3 minutes at 30fps
    if (!frameCount || frameCount > MAX_FRAMES) throw new Error('Video too long')

    const sha256 = precomputedSha256 || (await computeSha256(file)).sha256
    const created = await createJobForChecksum(sha256, ctx)
    if (created === 'skipped') {
        trackEvent('analysis_upload_skipped', { reason: 'duplicate_or_already_processed' })
        return 'skipped'
    }
    const job_id = created.job_id
    trackEvent('analysis_job_created', { job_id })

    const releaseVideoSlot = await acquireVideoUploadSlot()

    try {
        onStarted?.()
        const result = await uploadVideoFileToJob(file, quality, ctx, job_id, (percent: number) => {
            onProgress(percent)
            if (percent >= PERCENT_UPLOAD) releaseVideoSlot()
        })
        trackEvent('analysis_upload_complete', { job_id })
        return result
    } catch (e: any) {
        trackEvent('analysis_upload_failed', { job_id, message: String(e?.message || e || 'upload failed') })
        releaseVideoSlot()
        throw e
    }
}

export async function createJobForChecksum(
    sha256: string,
    ctx: UploadContext
): Promise<{ job_id: string } | 'skipped'> {
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
    const { job_id, status } = (await createRes.json()) as { job_id: string; status: string }
    // If the server indicates this checksum already has a non-pending job (e.g. succeeded),
    // treat it as a duplicate and skip uploading.
    if (status !== 'pending') return 'skipped'
    return { job_id }
}

export async function uploadVideoFileToJob(
    file: File,
    quality: UploadQuality,
    ctx: UploadContext,
    job_id: string,
    onProgress: (percent: number) => void
): Promise<'uploaded'> {
    // Step 2: Preprocess
    // TODO reenable: const processed = await preprocessVideo(file, quality, progress => onProgress(progress * PERCENT_PREPROCESS))
    const processed = await file.arrayBuffer()
    const totalSize = processed.byteLength
    const totalParts = Math.ceil(totalSize / CHUNK_SIZE)

    // Step 3: INIT chunked upload (also carries model params)
    const initForm = new FormData()
    initForm.append('total_size', String(totalSize))
    initForm.append('chunk_size', String(CHUNK_SIZE))
    initForm.append('total_parts', String(totalParts))
    initForm.append('file_name', file.name)
    initForm.append('mime_type', file.type || 'video/mp4')
    initForm.append('yolo_model', 'windsurfing/best.pt')

    const initRes = await fetch(`${MODAL_BASE}/jobs/${job_id}/upload/init`, {
        method: 'POST',
        headers: { Authorization: await ctx.getAuthHeader() },
        body: initForm,
    })
    if (!initRes.ok) throw new Error(await initRes.text())
    const { resume_from_part } = (await initRes.json()) as { resume_from_part: number }

    // Step 4: Upload parts (parallel with aggregated progress)
    const processedView = new Uint8Array(processed)

    // Precompute part sizes
    const partSizes: number[] = Array.from({ length: totalParts }, (_, i) => {
        const start = i * CHUNK_SIZE
        const end = Math.min(start + CHUNK_SIZE, totalSize)
        return end - start
    })

    // Bytes already safely on server due to resume
    const initialUploadedBytes = partSizes.slice(0, resume_from_part).reduce((a, b) => a + b, 0)

    // Track per-part uploaded bytes to aggregate progress across concurrent uploads
    const perPartUploaded: number[] = new Array(totalParts).fill(0)
    let aggregatedUploaded = initialUploadedBytes

    const updateOverallProgress = () => {
        const overall = aggregatedUploaded / totalSize
        onProgress(Math.min(PERCENT_UPLOAD, overall * PERCENT_UPLOAD + PERCENT_PREPROCESS))
    }

    // Work queue of part indices to upload
    const workIndices: number[] = []
    for (let i = resume_from_part; i < totalParts; i++) workIndices.push(i)

    let nextIdx = 0

    async function uploadOnePart(partIndex: number): Promise<void> {
        const start = partIndex * CHUNK_SIZE
        const end = Math.min(start + CHUNK_SIZE, totalSize)
        const chunkBytes = processedView.subarray(start, end)
        const partSize = partSizes[partIndex]

        const partForm = new FormData()
        partForm.append('part_index', String(partIndex))
        partForm.append(
            'chunk',
            new Blob([chunkBytes], { type: file.type || 'application/octet-stream' }),
            `${file.name}.part${partIndex}`
        )

        await doXhrUpload(
            `${MODAL_BASE}/jobs/${job_id}/upload/part`,
            ctx.getAuthHeader,
            partForm,
            (percent: number) => {
                const bytes = Math.max(0, Math.min(partSize, Math.round(percent * partSize)))
                const delta = bytes - perPartUploaded[partIndex]
                if (delta > 0) {
                    perPartUploaded[partIndex] += delta
                    aggregatedUploaded += delta
                    updateOverallProgress()
                }
            }
        )

        // Ensure we account for any rounding differences at completion
        const deltaToFull = partSize - perPartUploaded[partIndex]
        if (deltaToFull > 0) {
            perPartUploaded[partIndex] += deltaToFull
            aggregatedUploaded += deltaToFull
            updateOverallProgress()
        }
    }

    async function worker(): Promise<void> {
        while (true) {
            const i = nextIdx
            if (i >= workIndices.length) return
            nextIdx = i + 1
            const partIndex = workIndices[i]
            await uploadOnePart(partIndex)
        }
    }

    const concurrency = Math.min(MAX_CONCURRENCY, Math.max(1, workIndices.length))
    const workers = Array.from({ length: concurrency }, () => worker())
    await Promise.all(workers)

    // Step 5: COMPLETE upload (server concatenates, checksums, and spawns inference)
    const completeRes = await fetch(`${MODAL_BASE}/jobs/${job_id}/upload/complete`, {
        method: 'POST',
        headers: { Authorization: await ctx.getAuthHeader() },
    })
    if (!completeRes.ok) throw new Error(await completeRes.text())
    onProgress(1)
    return 'uploaded'
}
