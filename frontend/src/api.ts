import { auth, backendUrl } from './firebase'

export async function callBackend<T>(
    path: string,
    opts?: { method?: string; body?: unknown; forceRefreshToken?: boolean }
): Promise<T> {
    const user = auth.currentUser
    if (!user) throw new Error('Not signed in.')
    const token = await user.getIdToken(!!opts?.forceRefreshToken)

    const res = await fetch(`${backendUrl}${path}`, {
        method: opts?.method ?? 'GET',
        headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
        },
        body: opts?.body ? JSON.stringify(opts.body) : undefined,
    })

    const text = await res.text()
    const json = text ? (JSON.parse(text) as unknown) : null

    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}: ${JSON.stringify(json)}`)
    }

    return json as T
}

export async function callModal<T>(
    path: string,
    opts?: { method?: string; body?: BodyInit | null; headers?: Record<string, string>; forceRefreshToken?: boolean }
): Promise<T> {
    const user = auth.currentUser
    if (!user) throw new Error('Not signed in.')
    const token = await user.getIdToken(!!opts?.forceRefreshToken)

    // modal base is only used directly in the uploader; keep this helper for completeness.
    const res = await fetch(path, {
        method: opts?.method ?? 'GET',
        headers: {
            Authorization: `Bearer ${token}`,
            ...(opts?.headers ?? {}),
        },
        body: opts?.body ?? undefined,
    })

    const text = await res.text()
    const json = text ? (JSON.parse(text) as unknown) : null

    if (!res.ok) {
        throw new Error(`${res.status} ${res.statusText}: ${JSON.stringify(json)}`)
    }

    return json as T
}
