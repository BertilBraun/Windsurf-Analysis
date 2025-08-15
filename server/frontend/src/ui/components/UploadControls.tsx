import React from 'react'
import { useAuth } from '../auth/AuthProvider'
import { } from '../types'

export const UploadControls: React.FC<{ onSubmitted: (num: number) => void }> = ({ onSubmitted }) => {
    const { authorizedFetch } = useAuth()
    const [isUploading, setIsUploading] = React.useState(false)
    const [status, setStatus] = React.useState<string | null>(null)

    const computeSha256 = async (file: File) => {
        const arrayBuffer = await file.arrayBuffer()
        const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer)
        const hashArray = Array.from(new Uint8Array(hashBuffer))
        const sha256 = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
        return { arrayBuffer, sha256 }
    }

    const uploadSingle = async (file: File) => {
        const { arrayBuffer, sha256 } = await computeSha256(file)
        const form = new FormData()
        form.append('file', new Blob([arrayBuffer], { type: file.type }), file.name)
        form.append('original_file_path', file.name)
        form.append('original_checksum_sha256', sha256)
        form.append('yolo_model', 'windsurfing/2025_08_09_100epochs.pt')
        form.append('reid_model', 'common/osnet_ain_x1_0_msmt17.pth')
        await authorizedFetch('/jobs/upload', { method: 'POST', body: form })
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
                await uploadSingle(input.files[0])
                onSubmitted(1)
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
            ; (input as any).webkitdirectory = true
        input.multiple = true
        input.accept = 'video/*'
        input.onchange = async () => {
            if (!input.files || input.files.length === 0) return
            const files = Array.from(input.files).filter(f => f.type.startsWith('video/'))
            if (files.length === 0) return
            setIsUploading(true)
            setStatus(`Submitting ${files.length} jobs...`)
            try {
                await Promise.all(files.map(f => uploadSingle(f)))
                onSubmitted(files.length)
            } finally {
                setIsUploading(false)
                setStatus(null)
            }
        }
        input.click()
    }

    return (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <button onClick={onPickFile} disabled={isUploading}>{isUploading ? 'Submitting…' : 'Submit video'}</button>
            <button onClick={onPickFolder} disabled={isUploading}>{isUploading ? 'Submitting…' : 'Submit folder'}</button>
            {status && <span style={{ fontSize: 12, color: '#6b7280' }}>{status}</span>}
        </div>
    )
}


