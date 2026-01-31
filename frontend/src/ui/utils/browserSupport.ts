/**
 * @fileoverview Helpers for determining whether the current browser supports
 * the features required for the Full Analyzer experience.
 */

export type BrowserSupportReport = {
    supported: boolean
    missing: Array<'fileSystemAccess'>
}

const hasFileSystemAccessApi = () =>
    typeof window !== 'undefined' && 'showDirectoryPicker' in window

export function getAnalyzerBrowserSupport(): BrowserSupportReport {
    const missing: BrowserSupportReport['missing'] = []

    if (!hasFileSystemAccessApi()) missing.push('fileSystemAccess')

    return { supported: missing.length === 0, missing }
}
