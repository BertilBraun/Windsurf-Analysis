import React from 'react'
import { useAuth, API_BASE } from '../auth/AuthProvider'
import { preprocessVideo } from '../utils/preprocessVideo'

type UploadItem = {
    id: string
    name: string
    size: number
    progress: number
    status: 'pending' | 'hashing' | 'preflight' | 'preprocessing' | 'uploading' | 'done' | 'skipped' | 'error'
    error?: string
}

function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`
    const units = ['KB', 'MB', 'GB', 'TB']
    let b = bytes
    let i = 0
    while (b >= 1024 && i < units.length - 1) {
        b /= 1024
        i++
    }
    return `${b.toFixed(1)} ${units[i]}`
}

function UploadRow({ item }: { item: UploadItem }) {
    const statusText =
        item.status === 'hashing'
            ? 'Hashing…'
            : item.status === 'preflight'
            ? 'Checking…'
            : item.status === 'preprocessing'
            ? 'Preprocessing…'
            : item.status === 'uploading'
            ? `${item.progress}%`
            : item.status === 'done'
            ? 'Done'
            : item.status === 'skipped'
            ? 'Skipped (duplicate)'
            : item.status === 'error'
            ? 'Error'
            : ''
    const barColor = item.status === 'error' ? '#ef4444' : '#3b82f6'
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                <span
                    title={item.name}
                    style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 420 }}
                >
                    {item.name}
                </span>
                <span style={{ color: '#6b7280' }}>{statusText}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#6b7280' }}>
                <span>{formatBytes(item.size)}</span>
                <span>{item.status === 'uploading' ? `${item.progress}%` : ''}</span>
            </div>
            <div style={{ height: 8, background: '#eee', borderRadius: 4, overflow: 'hidden' }}>
                <div
                    style={{
                        width: `${item.progress}%`,
                        height: '100%',
                        background: barColor,
                        transition: 'width 120ms linear',
                    }}
                />
            </div>
            {item.error && <div style={{ color: '#ef4444', fontSize: 12 }}>{item.error}</div>}
        </div>
    )
}

function UploadList({ uploads }: { uploads: UploadItem[] }) {
    if (!uploads || uploads.length === 0) return null
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {uploads.map(u => (
                <UploadRow key={u.id} item={u} />
            ))}
        </div>
    )
}

function doXhrUpload(
    url: string,
    authHeader: string | null,
    form: FormData,
    onProgress: (percent: number) => void
): Promise<void> {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open('POST', url, true)
        if (authHeader) xhr.setRequestHeader('Authorization', authHeader)
        xhr.upload.onprogress = e => {
            if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
        }
        xhr.onerror = () => reject(new Error('Network error'))
        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) resolve()
            else reject(new Error(xhr.responseText || `HTTP ${xhr.status}`))
        }
        xhr.send(form)
    })
}

export const UploadControls: React.FC<{ onSubmitted: (num: number) => void }> = ({ onSubmitted }) => {
    const { authHeader, isAuthenticated, authorizedFetch } = useAuth()
    const [isUploading, setIsUploading] = React.useState(false)
    const [status, setStatus] = React.useState<string | null>(null)
    const [uploads, setUploads] = React.useState<UploadItem[]>([])
    const counterRef = React.useRef(1)

    const computeSha256 = async (file: File) => {
        const arrayBuffer = await file.arrayBuffer()
        const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer)
        const hashArray = Array.from(new Uint8Array(hashBuffer))
        const sha256 = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
        return { arrayBuffer, sha256 }
    }

    const uploadSingle = (file: File): Promise<boolean> => {
        const id = `${Date.now()}-${counterRef.current++}`
        setUploads(prev => [...prev, { id, name: file.name, size: file.size, progress: 0, status: 'hashing' }])
        return new Promise(async (resolve, reject) => {
            try {
                const { arrayBuffer, sha256 } = await computeSha256(file)
                // Preflight duplicate check
                setUploads(prev => prev.map(u => (u.id === id ? { ...u, status: 'preflight', progress: 0 } : u)))
                const res = await authorizedFetch('/videos/checksum', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ original_checksum_sha256: sha256 }),
                })
                const pre = (await res.json()) as { exists: boolean }
                if (pre.exists) {
                    setUploads(prev => prev.map(u => (u.id === id ? { ...u, status: 'skipped', progress: 100 } : u)))
                    resolve(false)
                    return
                }

                // Preprocess before upload
                setUploads(prev => prev.map(u => (u.id === id ? { ...u, status: 'preprocessing', progress: 0 } : u)))
                const processed = await preprocessVideo(file)

                // Build form and upload via XHR to capture progress
                setUploads(prev => prev.map(u => (u.id === id ? { ...u, status: 'uploading', progress: 0 } : u)))
                const form = new FormData()
                form.append('file', new Blob([processed.arrayBuffer], { type: processed.type }), processed.name)
                form.append('original_file_path', file.name)
                form.append('original_checksum_sha256', sha256)
                form.append('yolo_model', 'windsurfing/2025_08_09_100epochs.pt')
                form.append('reid_model', 'common/osnet_ain_x1_0_msmt17.pth')

                try {
                    await doXhrUpload(`${API_BASE}/jobs/upload`, authHeader, form, percent => {
                        setUploads(prev => prev.map(u => (u.id === id ? { ...u, progress: percent } : u)))
                    })
                    setUploads(prev => prev.map(u => (u.id === id ? { ...u, status: 'done', progress: 100 } : u)))
                    resolve(true)
                } catch (e: any) {
                    const msg = e?.message || 'Upload failed'
                    setUploads(prev => prev.map(u => (u.id === id ? { ...u, status: 'error', error: msg } : u)))
                    reject(new Error(msg))
                }
            } catch (err: any) {
                const msg = String(err?.message || 'Upload failed')
                setUploads(prev => prev.map(u => (u.id === id ? { ...u, status: 'error', error: msg } : u)))
                reject(err)
            }
        })
    }

    const onPickFile = async () => {
        const input = document.createElement('input')
        input.type = 'file'
        input.accept = 'video/*'
        input.onchange = async () => {
            if (!input.files || input.files.length === 0) return
            setIsUploading(true)
            setStatus('Submitting 1 job...')
            try {
                const uploaded = await uploadSingle(input.files[0])
                onSubmitted(uploaded ? 1 : 0)
            } finally {
                setIsUploading(false)
                setStatus(null)
            }
        }
        input.click()
    }

    const onPickFolder = async () => {
        const input = document.createElement('input')
        input.type = 'file'
        ;(input as any).webkitdirectory = true
        input.multiple = true
        input.accept = 'video/*'
        input.onchange = async () => {
            if (!input.files || input.files.length === 0) return
            const files = Array.from(input.files).filter(f => f.type.startsWith('video/'))
            if (files.length === 0) return
            setIsUploading(true)
            setStatus(`Submitting ${files.length} jobs...`)
            try {
                const results = await Promise.all(files.map(f => uploadSingle(f)))
                const uploadedCount = results.filter(Boolean).length
                onSubmitted(uploadedCount)
            } finally {
                setIsUploading(false)
                setStatus(null)
            }
        }
        input.click()
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <button onClick={onPickFile} disabled={isUploading || !isAuthenticated}>
                    {isUploading ? 'Submitting…' : 'Submit video'}
                </button>
                <button onClick={onPickFolder} disabled={isUploading || !isAuthenticated}>
                    {isUploading ? 'Submitting…' : 'Submit folder'}
                </button>
                {status && <span style={{ fontSize: 12, color: '#6b7280' }}>{status}</span>}
            </div>
            <UploadList uploads={uploads} />
        </div>
    )
}
