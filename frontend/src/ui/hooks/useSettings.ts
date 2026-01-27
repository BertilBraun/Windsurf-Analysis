/**
 * @module useSettings
 * Provides a hook for managing application settings with persistence in IndexedDB.
 */

import React from 'react'
import { loadSetting, saveSetting, deleteSetting } from '../utils/idb'
import { UploadQuality } from '../types'

/**
 * Keys used for storing settings in IndexedDB.
 */
export const SETTINGS_KEYS = {
    uploadQuality: 'uploadQuality',
    authEmail: 'authEmail',
    authHeader: 'authHeader',
} as const

/**
 * Represents the structure of the application settings state.
 */
export type SettingsState = {
    /** Preferred quality for media uploads. */
    uploadQuality: UploadQuality
    /** User email for authentication. */
    authEmail: string | null
    /** Authentication header string. */
    authHeader: string | null
}

/**
 * Hook to manage and persist application settings.
 *
 * Loads settings from IndexedDB on mount and provides methods to update them.
 *
 * @param initial - Optional initial settings to override defaults before IndexedDB loads.
 * @returns An object containing the current settings and functions to update them.
 */
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

    /**
     * Updates the upload quality setting and persists it to storage.
     */
    const setUploadQuality = React.useCallback(async (q: UploadQuality) => {
        setSettings(s => ({ ...s, uploadQuality: q }))
        await saveSetting(SETTINGS_KEYS.uploadQuality, q)
    }, [])

    /**
     * Updates authentication settings and persists them to storage.
     */
    const setAuth = React.useCallback(async (email: string, header: string) => {
        setSettings(s => ({ ...s, authEmail: email, authHeader: header }))
        await Promise.all([saveSetting(SETTINGS_KEYS.authEmail, email), saveSetting(SETTINGS_KEYS.authHeader, header)])
    }, [])

    /**
     * Clears authentication settings from state and persistent storage.
     */
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
