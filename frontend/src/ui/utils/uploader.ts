import { ALL_FORMATS, BlobSource, Input } from 'mediabunny'
import { UploadQuality } from '../types'
import { trackEvent } from './analytics'
import { auth, storage } from '../../firebase'
import { ref, uploadBytesResumable } from 'firebase/storage'

export type AuthorizedFetch = {
    (input: RequestInfo, init?: RequestInit): Promise<Response>
}

const MAX_PARALLEL_VIDEO_UPLOADS = 4
const MAX_FRAMES = 30 * 60 * 3 // 3 minutes at 30fps

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

const acquireVideoUploadSlot = createLimiter(MAX_PARALLEL_VIDEO_UPLOADS)

async function assertVideoWithinMaxFrames(file: File, maxFrames: number): Promise<void> {
    const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS })
    try {
        const videoTrack = await input.getPrimaryVideoTrack()
        if (!videoTrack) throw new Error('No video track found.')

        const duration = await videoTrack.computeDuration()
        const packetStats = await videoTrack.computePacketStats(100)
        const averageFrameRate = packetStats.averagePacketRate
        const totalFrames = duration * averageFrameRate
        if (totalFrames > maxFrames) throw new Error('Video too long')
    } finally {
        try {
            input.dispose()
        } catch {}
    }
}

// TODO with preprocess const PERCENT_PREPROCESS = 0.3
// TODO with preprocess const PERCENT_UPLOAD = 0.7
const PERCENT_PREPROCESS = 0.0
const PERCENT_UPLOAD = 0.95

export async function uploadVideoFile(params: {
    file: File
    quality: UploadQuality
    authorizedFetch: AuthorizedFetch
    onProgress: (percent: number) => void
    onStarted: () => void
    sha256: string
    existingJobId?: string
}): Promise<'uploaded' | 'skipped'> {
    const { file, quality, authorizedFetch, onProgress, onStarted, sha256, existingJobId } = params

    // Step 1: Create job (also acts as duplicate/quota check)
    trackEvent('analysis_upload_start', {
        file_size_bytes: file.size,
        mime: file.type || 'video/mp4',
        quality,
    })

    // Limit upload length by counting demuxed video samples.
    await assertVideoWithinMaxFrames(file, MAX_FRAMES)

    let job_id: string | null = null
    if (existingJobId) {
        job_id = existingJobId
        trackEvent('analysis_upload_resume', { job_id })
    } else {
        const created = await createJob(sha256, file, authorizedFetch)
        if (created === 'skipped') {
            trackEvent('analysis_upload_skipped', { reason: 'duplicate_or_already_processed' })
            return 'skipped'
        }
        trackEvent('analysis_job_created', { job_id: created.job_id })
        job_id = created.job_id
    }

    const releaseVideoSlot = await acquireVideoUploadSlot()

    try {
        onStarted?.()
        const result = await uploadVideoFileToJob(file, quality, authorizedFetch, job_id, (percent: number) => {
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

export async function createJob(
    sha256: string,
    file: File,
    authorizedFetch: AuthorizedFetch
): Promise<{ job_id: string } | 'skipped'> {
    const createRes = await authorizedFetch('/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            original_checksum_sha256: sha256,
            original_file_size_bytes: file.size,
            original_file_mime_type: file.type || 'video/mp4',
        }),
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
    // If the server indicates this checksum already has a non-uploading job (e.g. succeeded),
    // treat it as a duplicate and skip uploading.
    if (status !== 'uploading') return 'skipped'
    return { job_id }
}

async function uploadVideoToFirebaseStorage(params: {
    file: File
    job_id: string
    onProgress: (percent: number) => void
}): Promise<{ object_path: string }> {
    const { file, job_id, onProgress } = params
    const user = auth.currentUser
    if (!user) throw new Error('Not authenticated')

    const object_path = `uploads/${user.uid}/${job_id}.mp4`
    const uploadRef = ref(storage, object_path)

    await new Promise<void>((resolve, reject) => {
        const task = uploadBytesResumable(uploadRef, file, {
            contentType: file.type || 'video/mp4',
        })

        task.on(
            'state_changed',
            snap => {
                onProgress(
                    Math.min(
                        PERCENT_UPLOAD,
                        (snap.bytesTransferred / snap.totalBytes) * PERCENT_UPLOAD + PERCENT_PREPROCESS
                    )
                )
            },
            err => reject(err),
            () => resolve()
        )
    })

    return { object_path }
}

export async function uploadVideoFileToJob(
    file: File,
    quality: UploadQuality,
    authorizedFetch: AuthorizedFetch,
    job_id: string,
    onProgress: (percent: number) => void
): Promise<'uploaded'> {
    // Step 2: Preprocess
    // TODO reenable: const processed = await preprocessVideo(file, quality, progress => onProgress(progress * PERCENT_PREPROCESS))
    const { object_path } = await uploadVideoToFirebaseStorage({ file, job_id, onProgress })

    // Step 3: Mark upload complete and start processing
    const completeRes = await authorizedFetch(`/jobs/${job_id}/upload/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            object_path,
            size_bytes: file.size,
            mime_type: file.type || 'video/mp4',
            yolo_model: 'windsurfing_pose/best.pt',
        }),
    })
    if (!completeRes.ok) throw new Error(await completeRes.text())
    onProgress(1)
    return 'uploaded'
}
