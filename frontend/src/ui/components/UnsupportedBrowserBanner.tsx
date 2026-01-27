/**
 * @fileoverview Provides a banner component to warn users when their browser
 * lacks required features like WebCodecs or File System Access API.
 */

import React from 'react'
import { useTranslation } from 'react-i18next'

const hasWebCodecs = () =>
    'VideoDecoder' in window && 'EncodedVideoChunk' in window && 'VideoFrame' in window

const hasFolderPicker = () => 'showDirectoryPicker' in window

/**
 * A banner component that conditionally renders a warning if the browser does
 * not support necessary APIs (WebCodecs and Directory Picker).
 */
export const UnsupportedBrowserBanner: React.FC = () => {
    const { t } = useTranslation()
    const [isUnsupported, setIsUnsupported] = React.useState(false)

    React.useEffect(() => {
        setIsUnsupported(!hasWebCodecs() || !hasFolderPicker())
    }, [])

    if (!isUnsupported) return null

    return (
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-3 text-amber-900">
            <div className="mx-auto flex max-w-[1400px] flex-col gap-1 text-sm">
                <div className="font-semibold">{t('components.unsupportedBrowser.title')}</div>
                <div className="text-xs">{t('components.unsupportedBrowser.body')}</div>
            </div>
        </div>
    )
}
