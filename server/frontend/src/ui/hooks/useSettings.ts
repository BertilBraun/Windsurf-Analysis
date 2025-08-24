import React from 'react'
import { loadSetting, saveSetting, deleteSetting } from '../utils/idb'

export type UploadQuality = 'original' | 'high' | 'medium' | 'minimum'

export const SETTINGS_KEYS = {
    uploadQuality: 'uploadQuality',
    authEmail: 'authEmail',
    authHeader: 'authHeader',
} as const

export type SettingsState = {
    uploadQuality: UploadQuality
    authEmail: string | null
    authHeader: string | null
}

export function useSettings(initial?: Partial<SettingsState>) {
    const [settings, setSettings] = React.useState<SettingsState>({
        uploadQuality: initial?.uploadQuality ?? 'medium',
        authEmail: initial?.authEmail ?? null,
        authHeader: initial?.authHeader ?? null,
    })

    React.useEffect(() => {
        ;(async () => {
            const [uq, ae, ah] = await Promise.all([
                loadSetting<UploadQuality>(SETTINGS_KEYS.uploadQuality),
                loadSetting<string>(SETTINGS_KEYS.authEmail),
                loadSetting<string>(SETTINGS_KEYS.authHeader),
            ])
            setSettings(s => ({
                ...s,
                uploadQuality: uq ?? s.uploadQuality ?? 'medium',
                authEmail: ae ?? null,
                authHeader: ah ?? null,
            }))
        })()
        return () => {}
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const setUploadQuality = React.useCallback(async (q: UploadQuality) => {
        setSettings(s => ({ ...s, uploadQuality: q }))
        await saveSetting(SETTINGS_KEYS.uploadQuality, q)
    }, [])

    const setAuth = React.useCallback(async (email: string, header: string) => {
        setSettings(s => ({ ...s, authEmail: email, authHeader: header }))
        await Promise.all([saveSetting(SETTINGS_KEYS.authEmail, email), saveSetting(SETTINGS_KEYS.authHeader, header)])
    }, [])

    const clearAuth = React.useCallback(async () => {
        setSettings(s => ({ ...s, authEmail: null, authHeader: null }))
        await Promise.all([deleteSetting(SETTINGS_KEYS.authEmail), deleteSetting(SETTINGS_KEYS.authHeader)])
    }, [])

    return {
        settings,
        setUploadQuality,
        setAuth,
        clearAuth,
    }
}
