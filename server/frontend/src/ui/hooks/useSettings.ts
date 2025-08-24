import React from 'react'
import { loadSetting, saveSetting } from '../utils/idb'

export type UploadQuality = 'original' | 'high' | 'medium' | 'minimum'

export const SETTINGS_KEYS = {
    uploadQuality: 'uploadQuality',
} as const

export type SettingsState = {
    uploadQuality: UploadQuality
}

export function useSettings(initial?: Partial<SettingsState>) {
    const [settings, setSettings] = React.useState<SettingsState>({
        uploadQuality: initial?.uploadQuality ?? 'medium',
    })

    React.useEffect(() => {
        loadSetting(SETTINGS_KEYS.uploadQuality).then(q => setSettings(s => ({ ...s, uploadQuality: q ?? 'medium' })))
        return () => {}
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const setUploadQuality = React.useCallback(async (q: UploadQuality) => {
        setSettings(s => ({ ...s, uploadQuality: q }))
        await saveSetting(SETTINGS_KEYS.uploadQuality, q)
    }, [])

    return {
        settings,
        setUploadQuality,
    }
}
