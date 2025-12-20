import React from 'react'
import { useTranslation } from 'react-i18next'
import { JobSummary } from '../types'
import { AnimatedDots } from './AnimatedDots'
import JobThumbnail from './JobThumbnail'
import { trackEvent } from '../utils/analytics'

export type JobListSortKey = 'name' | 'date'
export type JobListSortDir = 'asc' | 'desc'

type FolderNode = {
    name: string
    path: string
    children: Map<string, FolderNode>
    jobs: JobInstance[]
    totalJobs: number
    activeJobs: number
}

type JobInstance = JobSummary & {
    // For display purposes we render one tile per local file path.
    local_relative_path: string
}

function normalizeRelativePath(path: string): string {
    return String(path || '')
        .replace(/^[./\\]+/, '')
        .replace(/\\/g, '/')
}

function splitPathParts(path: string): string[] {
    return normalizeRelativePath(path)
        .split('/')
        .map(p => p.trim())
        .filter(Boolean)
}

function basename(path: string | null | undefined): string {
    const parts = path ? splitPathParts(path) : []
    return parts.length ? parts[parts.length - 1] : ''
}

function stripMp4(name: string): string {
    return name.replace(/\.mp4$/i, '')
}

function sortJobs(list: JobInstance[], sortKey: JobListSortKey, sortDir: JobListSortDir): JobInstance[] {
    const out = [...list]
    out.sort((a, b) => {
        let cmp = 0
        if (sortKey === 'date') {
            const ta = Date.parse(a.updated_at || a.created_at || '') || 0
            const tb = Date.parse(b.updated_at || b.created_at || '') || 0
            cmp = ta < tb ? -1 : ta > tb ? 1 : 0
        } else if (sortKey === 'name') {
            // Within a folder, sort by filename; fallback to full relative path / id
            const an =
                stripMp4(basename(a.local_relative_path)).toLowerCase() ||
                normalizeRelativePath(a.local_relative_path || '').toLowerCase() ||
                a.id
            const bn =
                stripMp4(basename(b.local_relative_path)).toLowerCase() ||
                normalizeRelativePath(b.local_relative_path || '').toLowerCase() ||
                b.id
            cmp = an < bn ? -1 : an > bn ? 1 : 0
        } else {
            throw new Error(`Unknown sort key: ${sortKey}`)
        }
        return sortDir === 'asc' ? cmp : -cmp
    })
    return out
}

function sortedFolderNames(node: FolderNode): string[] {
    return Array.from(node.children.keys()).sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()))
}

function buildJobTree(jobs: JobSummary[]) {
    const root: FolderNode = {
        name: '',
        path: '',
        children: new Map(),
        jobs: [],
        totalJobs: 0,
        activeJobs: 0,
    }
    const unmappedJobs: JobSummary[] = []

    const ensureChild = (parent: FolderNode, childName: string, childPath: string): FolderNode => {
        const existing = parent.children.get(childName)
        if (existing) return existing
        const created: FolderNode = {
            name: childName,
            path: childPath,
            children: new Map(),
            jobs: [],
            totalJobs: 0,
            activeJobs: 0,
        }
        parent.children.set(childName, created)
        return created
    }

    const expandToInstances = (job: JobSummary): JobInstance[] => {
        const rels = job.local_relative_paths && job.local_relative_paths.length ? job.local_relative_paths : null
        const candidates = rels?.length ? rels : job.local_relative_path ? [job.local_relative_path] : []
        return candidates
            .map(rel => normalizeRelativePath(rel))
            .filter(Boolean)
            .map(rel => ({ ...job, local_relative_path: rel }))
    }

    for (const job of jobs) {
        const instances = expandToInstances(job)
        if (instances.length === 0) {
            unmappedJobs.push(job)
            continue
        }

        for (const inst of instances) {
            const parts = splitPathParts(inst.local_relative_path)
            if (parts.length === 0) continue
            const dirParts = parts.slice(0, Math.max(0, parts.length - 1))
            let node = root
            let currentPath = ''
            for (const part of dirParts) {
                currentPath = currentPath ? `${currentPath}/${part}` : part
                node = ensureChild(node, part, currentPath)
            }
            node.jobs.push(inst)
        }
    }

    const isActive = (s: JobSummary['status']) => s !== 'succeeded' && s !== 'failed' && s !== 'canceled'

    const finalize = (node: FolderNode): { total: number; active: number } => {
        let total = node.jobs.length
        let active = node.jobs.reduce((sum, j) => sum + (isActive(j.status) ? 1 : 0), 0)
        for (const child of node.children.values()) {
            const res = finalize(child)
            total += res.total
            active += res.active
        }
        node.totalJobs = total
        node.activeJobs = active
        return { total, active }
    }
    finalize(root)

    // Collect all folder paths (for expand/collapse all)
    const folderPaths: string[] = []
    const collect = (node: FolderNode) => {
        for (const child of node.children.values()) {
            folderPaths.push(child.path)
            collect(child)
        }
    }
    collect(root)

    return { root, folderPaths, unmappedJobs }
}

export function getJobListOrderedJobIds(
    jobs: JobSummary[],
    sortKey: JobListSortKey,
    sortDir: JobListSortDir
): string[] {
    const { root } = buildJobTree(jobs)
    const ordered: JobInstance[] = []

    const walk = (node: FolderNode) => {
        for (const childName of sortedFolderNames(node)) {
            const child = node.children.get(childName)
            if (child) walk(child)
        }
        for (const j of sortJobs(node.jobs, sortKey, sortDir)) ordered.push(j)
    }

    walk(root)
    // Note: duplicates (multiple local paths for the same job id) are represented multiple times.
    return ordered.map(j => j.id)
}

export const JobList: React.FC<{
    jobs: JobSummary[]
    sortKey: JobListSortKey
    sortDir: JobListSortDir
    onToggleSort: (key: JobListSortKey) => void
    onOpen: (id: string, localRelativePath?: string | null) => void
    openingId?: string | null
    dirHandle?: FileSystemDirectoryHandle | null
    initialSyncComplete?: boolean
}> = ({ jobs, sortKey, sortDir, onToggleSort, onOpen, openingId, dirHandle = null, initialSyncComplete = false }) => {
    const { t } = useTranslation()
    const [expanded, setExpanded] = React.useState<Set<string>>(() => new Set(['']))

    const { root, folderPaths, unmappedJobs } = React.useMemo(() => buildJobTree(jobs), [jobs])

    const toggleFolder = React.useCallback((path: string) => {
        setExpanded(prev => {
            const next = new Set(prev)
            if (next.has(path)) next.delete(path)
            else next.add(path)
            next.add('') // root always expanded
            return next
        })
    }, [])

    const expandAll = React.useCallback(() => {
        setExpanded(new Set(['', ...folderPaths]))
    }, [folderPaths])

    const collapseAll = React.useCallback(() => {
        setExpanded(new Set(['']))
    }, [])

    const renderFolder = React.useCallback(
        (node: FolderNode, depth: number) => {
            const isRoot = node.path === ''
            const isOpen = expanded.has(node.path) || isRoot
            const childNames = sortedFolderNames(node)
            const hasChildren = childNames.length > 0
            const hasJobs = node.jobs.length > 0
            const isProcessing = node.activeJobs > 0

            // Hide an empty root header; everything else gets a header row.
            const showHeader = !isRoot

            return (
                <div key={node.path || 'root'} className="flex flex-col">
                    {showHeader && (
                        <button
                            type="button"
                            onClick={() => {
                                trackEvent('joblist_folder_toggle')
                                toggleFolder(node.path)
                            }}
                            className="flex items-center gap-2 text-left py-1.5 rounded-md hover:bg-slate-50"
                            style={{ paddingLeft: depth * 12 }}
                            title={node.path}
                        >
                            <span className="w-4 text-slate-500">
                                {hasChildren || hasJobs ? (isOpen ? '▾' : '▸') : ''}
                            </span>
                            <span className="font-medium text-slate-900 flex items-center gap-2">
                                {node.name}
                                {isProcessing && (
                                    <span
                                        className="inline-flex items-center gap-1 text-[11px] text-blue-700"
                                        title={t('components.jobList.folder.inProgress', { count: node.activeJobs })}
                                    >
                                        <span className="inline-block w-3 h-3 rounded-full border-2 border-blue-600/30 border-t-blue-600 animate-spin" />
                                    </span>
                                )}
                            </span>
                            <span className="text-xs text-slate-500">({node.totalJobs})</span>
                        </button>
                    )}

                    {isOpen && (
                        <>
                            {childNames.map(childName => {
                                const child = node.children.get(childName)!
                                return renderFolder(child, depth + 1)
                            })}

                            {node.jobs.length > 0 && (
                                <div
                                    className="mt-2 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"
                                    style={{ paddingLeft: (depth + (showHeader ? 1 : 0)) * 12 }}
                                >
                                    {sortJobs(node.jobs, sortKey, sortDir).map(job => {
                                        const fileName =
                                            stripMp4(basename(job.local_relative_path)) || t('common.notAvailable')
                                        const caption = fileName
                                        return (
                                            <div
                                                key={`${job.id}::${job.local_relative_path ?? ''}`}
                                                className="flex flex-col items-start"
                                            >
                                                <div
                                                    className={`relative ${
                                                        job.status === 'succeeded' ? 'cursor-pointer' : 'cursor-default'
                                                    } ${openingId === job.id ? 'opacity-60' : ''}`}
                                                    onClick={() => {
                                                        if (job.status === 'succeeded' && job.local_relative_path)
                                                            onOpen(job.id, job.local_relative_path)
                                                    }}
                                                    role="button"
                                                    tabIndex={0}
                                                    onKeyDown={e =>
                                                        job.status === 'succeeded' &&
                                                        (e.key === 'Enter' || e.key === ' ') &&
                                                        onOpen(job.id, job.local_relative_path)
                                                    }
                                                >
                                                    <JobThumbnail job={job} dirHandle={dirHandle} />
                                                    {openingId === job.id && (
                                                        <div className="absolute inset-0 flex items-center justify-center">
                                                            <div className="text-white text-sm bg-black/60 rounded px-2 py-1">
                                                                {t('components.jobList.opening')}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                                <div
                                                    className="mt-1 max-w-48 truncate text-xs text-gray-700"
                                                    title={caption}
                                                >
                                                    {caption}
                                                </div>
                                            </div>
                                        )
                                    })}
                                </div>
                            )}
                        </>
                    )}
                    {showHeader && <div className="mt-3 border-t border-slate-200" />}
                </div>
            )
        },
        [dirHandle, expanded, onOpen, openingId, sortDir, sortKey, t, toggleFolder]
    )

    if (jobs.length === 0) {
        return (
            <div className="text-center text-gray-500">
                {initialSyncComplete ? t('components.jobList.waiting') : t('components.jobList.loading')}
                <AnimatedDots />
            </div>
        )
    }

    return (
        <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
                <div className="text-sm text-gray-600">{t('components.jobList.sort.label')}</div>
                <div className="flex gap-2 items-center">
                    <button
                        className={`px-2 py-1 rounded-md text-sm border ${
                            sortKey === 'name'
                                ? 'bg-gray-700 text-gray-100 border-gray-700'
                                : 'bg-gray-100 text-gray-800 border-gray-300'
                        }`}
                        onClick={() => {
                            trackEvent('joblist_sort', { key: 'name' })
                            onToggleSort('name')
                        }}
                    >
                        {t('components.jobList.sort.name')} {sortKey === 'name' ? (sortDir === 'asc' ? '▲' : '▼') : '↕'}
                    </button>
                    <button
                        className={`px-2 py-1 rounded-md text-sm border ${
                            sortKey === 'date'
                                ? 'bg-gray-700 text-gray-100 border-gray-700'
                                : 'bg-gray-100 text-gray-800 border-gray-300'
                        }`}
                        onClick={() => {
                            trackEvent('joblist_sort', { key: 'date' })
                            onToggleSort('date')
                        }}
                    >
                        {t('components.jobList.sort.date')} {sortKey === 'date' ? (sortDir === 'asc' ? '▲' : '▼') : '↕'}
                    </button>
                    <div className="w-px h-6 bg-slate-200 mx-1" />
                    <button
                        className="px-2 py-1 rounded-md text-sm border bg-gray-100 text-gray-800 border-gray-300 hover:bg-gray-200"
                        onClick={() => {
                            trackEvent('joblist_expand_all')
                            expandAll()
                        }}
                        title={t('components.jobList.actions.expandTitle')}
                    >
                        {t('components.jobList.actions.expand')}
                    </button>
                    <button
                        className="px-2 py-1 rounded-md text-sm border bg-gray-100 text-gray-800 border-gray-300 hover:bg-gray-200"
                        onClick={() => {
                            trackEvent('joblist_collapse_all')
                            collapseAll()
                        }}
                        title={t('components.jobList.actions.collapseTitle')}
                    >
                        {t('components.jobList.actions.collapse')}
                    </button>
                </div>
            </div>
            {/* Unmapped jobs (no known local path) */}
            {unmappedJobs.length > 0 && (
                <div className="rounded-md border border-amber-200 bg-amber-50 p-3">
                    <div className="text-xs font-semibold text-amber-900 mb-2">
                        {t('components.jobList.unmapped.title')}
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {[...unmappedJobs]
                            .sort((a, b) => {
                                if (sortKey === 'date') {
                                    const ta = Date.parse(a.updated_at || a.created_at || '') || 0
                                    const tb = Date.parse(b.updated_at || b.created_at || '') || 0
                                    return sortDir === 'asc' ? ta - tb : tb - ta
                                }
                                // name
                                const an =
                                    stripMp4(basename(a.local_relative_path)).toLowerCase() ||
                                    normalizeRelativePath(a.local_relative_path || '').toLowerCase() ||
                                    a.id
                                const bn =
                                    stripMp4(basename(b.local_relative_path)).toLowerCase() ||
                                    normalizeRelativePath(b.local_relative_path || '').toLowerCase() ||
                                    b.id
                                if (an === bn) return 0
                                return sortDir === 'asc' ? (an < bn ? -1 : 1) : an < bn ? 1 : -1
                            })
                            .map(job => {
                                const caption = stripMp4(basename(job.local_relative_path)) || t('common.notAvailable')
                                return (
                                    <div key={job.id} className="flex flex-col items-start">
                                        <div
                                            className={`relative ${
                                                job.status === 'succeeded' ? 'cursor-pointer' : 'cursor-default'
                                            } ${openingId === job.id ? 'opacity-60' : ''}`}
                                            onClick={() => {
                                                if (job.status === 'succeeded' && job.local_relative_path)
                                                    onOpen(job.id, job.local_relative_path)
                                            }}
                                            role="button"
                                            tabIndex={0}
                                            onKeyDown={e =>
                                                job.status === 'succeeded' &&
                                                (e.key === 'Enter' || e.key === ' ') &&
                                                onOpen(job.id, job.local_relative_path)
                                            }
                                        >
                                            <JobThumbnail job={job} dirHandle={dirHandle} />
                                            {openingId === job.id && (
                                                <div className="absolute inset-0 flex items-center justify-center">
                                                    <div className="text-white text-sm bg-black/60 rounded px-2 py-1">
                                                        {t('components.jobList.opening')}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                        <div className="mt-1 max-w-48 truncate text-xs text-gray-700" title={caption}>
                                            {caption}
                                        </div>
                                    </div>
                                )
                            })}
                    </div>
                </div>
            )}

            <div className="flex flex-col gap-1">{renderFolder(root, 0)}</div>
        </div>
    )
}
