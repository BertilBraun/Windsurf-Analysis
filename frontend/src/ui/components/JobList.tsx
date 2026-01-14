import React from 'react'
import { useTranslation } from 'react-i18next'
import { JobSummary } from '../types'
import { AnimatedDots } from './AnimatedDots'
import { Button } from './Button'
import JobThumbnail from './JobThumbnail'
import { Modal } from './Modal'
import { trackEvent } from '../utils/analytics'
import trashbinSvg from '../assets/trashbin.svg'
import { Spinner } from './Spinner'

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

type SortElement = {
    id: string
    updated_at?: string | null
    created_at?: string | null
    local_relative_path?: string | null
}

function _sortCriteria(a: SortElement, b: SortElement, sortKey: JobListSortKey): number {
    if (sortKey === 'date') {
        const ta = Date.parse(a.updated_at || a.created_at || '') || 0
        const tb = Date.parse(b.updated_at || b.created_at || '') || 0
        return ta < tb ? -1 : ta > tb ? 1 : 0
    } else if (sortKey === 'name') {
        const aPath = a.local_relative_path || ''
        const bPath = b.local_relative_path || ''
        const an = stripMp4(basename(aPath)).toLowerCase() || normalizeRelativePath(aPath).toLowerCase() || a.id
        const bn = stripMp4(basename(bPath)).toLowerCase() || normalizeRelativePath(bPath).toLowerCase() || b.id
        return an < bn ? -1 : an > bn ? 1 : 0
    } else {
        throw new Error(`Unknown sort key: ${sortKey}`)
    }
}

function _sortCompare(a: SortElement, b: SortElement, sortKey: JobListSortKey, sortDir: JobListSortDir): number {
    const cmp = _sortCriteria(a, b, sortKey)
    return sortDir === 'asc' ? cmp : -cmp
}

function sortJobs(list: JobInstance[], sortKey: JobListSortKey, sortDir: JobListSortDir): JobInstance[] {
    const out = [...list]
    out.sort((a, b) => _sortCompare(a, b, sortKey, sortDir))
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
        let active = node.jobs.filter(j => isActive(j.status)).length
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

const TrashIcon: React.FC<{ className?: string; tone?: 'default' | 'danger' }> = ({ className, tone = 'default' }) => {
    const filter =
        tone === 'danger'
            ? 'invert(21%) sepia(92%) saturate(7440%) hue-rotate(356deg) brightness(97%) contrast(117%)'
            : 'none'
    return <img src={trashbinSvg} alt="" className={`block h-4 w-4 ${className ?? ''}`} style={{ filter }} />
}

const SortBar: React.FC<{
    sortKey: JobListSortKey
    sortDir: JobListSortDir
    onToggleSort: (key: JobListSortKey) => void
    onExpandAll: () => void
    onCollapseAll: () => void
}> = ({ sortKey, sortDir, onToggleSort, onExpandAll, onCollapseAll }) => {
    const { t } = useTranslation()

    return (
        <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">{t('components.jobList.sort.label')}</div>
            <div className="flex gap-2 items-center">
                <Button
                    variant={sortKey === 'name' ? 'secondary' : 'outline'}
                    size="sm"
                    onClick={() => {
                        trackEvent('joblist_sort', { key: 'name' })
                        onToggleSort('name')
                    }}
                >
                    {t('components.jobList.sort.name')}{' '}
                    {sortKey === 'name'
                        ? sortDir === 'asc'
                            ? t('components.jobList.sort.ascSymbol')
                            : t('components.jobList.sort.descSymbol')
                        : t('components.jobList.sort.noneSymbol')}
                </Button>
                <Button
                    variant={sortKey === 'date' ? 'secondary' : 'outline'}
                    size="sm"
                    onClick={() => {
                        trackEvent('joblist_sort', { key: 'date' })
                        onToggleSort('date')
                    }}
                >
                    {t('components.jobList.sort.date')}{' '}
                    {sortKey === 'date'
                        ? sortDir === 'asc'
                            ? t('components.jobList.sort.ascSymbol')
                            : t('components.jobList.sort.descSymbol')
                        : t('components.jobList.sort.noneSymbol')}
                </Button>
                <div className="w-px h-6 bg-slate-200 mx-1" />
                <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                        trackEvent('joblist_expand_all')
                        onExpandAll()
                    }}
                    title={t('components.jobList.actions.expandTitle')}
                >
                    {t('components.jobList.actions.expand')}
                </Button>
                <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                        trackEvent('joblist_collapse_all')
                        onCollapseAll()
                    }}
                    title={t('components.jobList.actions.collapseTitle')}
                >
                    {t('components.jobList.actions.collapse')}
                </Button>
            </div>
        </div>
    )
}

const DeleteAllModal: React.FC<{
    open: boolean
    count: number
    bulkDeleting: boolean
    onClose: () => void
    onConfirm: () => void
}> = ({ open, count, bulkDeleting, onClose, onConfirm }) => {
    const { t } = useTranslation()

    if (!open) return null

    return (
        <Modal onClose={onClose} title={t('components.jobList.unmapped.deleteAll.modalTitle')}>
            <div className="p-4 text-sm text-slate-700">
                {t('components.jobList.unmapped.deleteAll.modalBody', { count })}
            </div>
            <div className="px-4 pb-4 flex items-center justify-end gap-2">
                <Button variant="ghost" onClick={onClose} text={t('common.cancel')} />
                <Button
                    variant="danger"
                    onClick={onConfirm}
                    text={t('components.jobList.unmapped.deleteAll.confirm')}
                    isPending={bulkDeleting}
                />
            </div>
        </Modal>
    )
}

const UnmappedJobsSection: React.FC<{
    unmappedJobs: JobSummary[]
    sortKey: JobListSortKey
    sortDir: JobListSortDir
    dirHandle: FileSystemDirectoryHandle | null
    onDeleteJobs: (ids: string[]) => Promise<number>
}> = ({ unmappedJobs, sortKey, sortDir, dirHandle, onDeleteJobs }) => {
    const { t } = useTranslation()
    const [isOpen, setIsOpen] = React.useState<boolean>(true)
    const [pendingDeleteId, setPendingDeleteId] = React.useState<string | null>(null)
    const [deletingJobIds, setDeletingJobIds] = React.useState<Set<string>>(() => new Set())
    const [showDeleteAllModal, setShowDeleteAllModal] = React.useState<boolean>(false)
    const [bulkDeleting, setBulkDeleting] = React.useState<boolean>(false)
    const pendingDeleteTimeoutRef = React.useRef<number | null>(null)
    const prevCountRef = React.useRef<number>(unmappedJobs.length)

    const sortedUnmappedJobs = React.useMemo(() => {
        return [...unmappedJobs].sort((a, b) => _sortCompare(a, b, sortKey, sortDir))
    }, [unmappedJobs, sortDir, sortKey])

    React.useEffect(() => {
        if (unmappedJobs.length > 0 && prevCountRef.current === 0) setIsOpen(true)
        prevCountRef.current = unmappedJobs.length
    }, [unmappedJobs.length])

    const clearPendingDeleteTimeout = React.useCallback(() => {
        if (pendingDeleteTimeoutRef.current) {
            window.clearTimeout(pendingDeleteTimeoutRef.current)
            pendingDeleteTimeoutRef.current = null
        }
    }, [])

    const clearPendingDelete = React.useCallback(() => {
        clearPendingDeleteTimeout()
        setPendingDeleteId(null)
    }, [clearPendingDeleteTimeout])

    React.useEffect(() => {
        return () => {
            clearPendingDeleteTimeout()
        }
    }, [clearPendingDeleteTimeout])

    const markDeleting = React.useCallback((id: string) => {
        setDeletingJobIds(prev => {
            const next = new Set(prev)
            next.add(id)
            return next
        })
    }, [])

    const unmarkDeleting = React.useCallback((id: string) => {
        setDeletingJobIds(prev => {
            const next = new Set(prev)
            next.delete(id)
            return next
        })
    }, [])

    const handleDeleteJob = React.useCallback(
        async (id: string) => {
            markDeleting(id)
            try {
                await onDeleteJobs([id])
                trackEvent('joblist_unmapped_delete', { job_id: id })
            } catch (e) {
                console.error('Failed to delete job', e)
            } finally {
                unmarkDeleting(id)
            }
        },
        [markDeleting, onDeleteJobs, unmarkDeleting]
    )

    const armPendingDelete = React.useCallback(
        (id: string) => {
            clearPendingDeleteTimeout()
            setPendingDeleteId(id)
            pendingDeleteTimeoutRef.current = window.setTimeout(() => {
                setPendingDeleteId(prev => (prev === id ? null : prev))
            }, 3000)
        },
        [clearPendingDeleteTimeout]
    )

    const handleDeleteClick = React.useCallback(
        (id: string) => {
            if (bulkDeleting || deletingJobIds.has(id)) return
            if (pendingDeleteId === id) {
                clearPendingDelete()
                void handleDeleteJob(id)
                return
            }
            armPendingDelete(id)
        },
        [armPendingDelete, bulkDeleting, clearPendingDelete, deletingJobIds, handleDeleteJob, pendingDeleteId]
    )

    const handleDeleteAll = React.useCallback(async () => {
        if (bulkDeleting || unmappedJobs.length === 0) return
        setShowDeleteAllModal(false)
        clearPendingDelete()
        setBulkDeleting(true)
        setDeletingJobIds(new Set(unmappedJobs.map(job => job.id)))
        try {
            await onDeleteJobs(unmappedJobs.map(j => j.id))
            trackEvent('joblist_unmapped_delete_all', { count: unmappedJobs.length })
        } catch (e) {
            console.error('Failed to delete all unmapped jobs', e)
        } finally {
            setBulkDeleting(false)
            setDeletingJobIds(new Set())
        }
    }, [bulkDeleting, clearPendingDelete, unmappedJobs, onDeleteJobs])

    const toggleOpen = React.useCallback(() => {
        setIsOpen(prev => {
            const next = !prev
            trackEvent('joblist_unmapped_toggle', { open: next })
            return next
        })
    }, [])

    if (unmappedJobs.length === 0) return null

    if (!isOpen) {
        return (
            <Button
                type="button"
                variant="warning"
                size="sm"
                onClick={toggleOpen}
                title={t('components.jobList.unmapped.toggleTitle')}
                className="justify-start"
            >
                <span className="w-4 text-amber-700">{t('components.jobList.folder.closedIcon')}</span>
                <span>{t('components.jobList.unmapped.collapsedLabel', { count: unmappedJobs.length })}</span>
                <div className="flex-1" />
            </Button>
        )
    }

    return (
        <div className="rounded-md border border-amber-200 bg-amber-50 p-3">
            <div className="flex items-start justify-between gap-3 mb-2">
                <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={toggleOpen}
                    title={t('components.jobList.unmapped.collapseTitle')}
                    className="justify-start text-amber-900"
                >
                    <span className="w-4 text-amber-700">{t('components.jobList.folder.openIcon')}</span>
                    <span>{t('components.jobList.unmapped.title', { count: unmappedJobs.length })}</span>
                </Button>
                <Button
                    variant="ghost"
                    size="sm"
                    className="text-xs"
                    onClick={() => {
                        trackEvent('joblist_unmapped_delete_all_open')
                        setShowDeleteAllModal(true)
                    }}
                    title={t('components.jobList.unmapped.deleteAll.title')}
                    disabled={bulkDeleting}
                    text={t('components.jobList.unmapped.deleteAll.label')}
                />
            </div>
            <div className="mb-3 text-xs text-amber-900/90">
                <div className="font-semibold">
                    {t('components.analyzerTutorialModal.steps.open.troubleshoot.title')}
                </div>
                <ul className="mt-1 list-disc pl-5 space-y-1">
                    <li>{t('components.analyzerTutorialModal.steps.open.troubleshoot.bullets.folder')}</li>
                    <li>{t('components.analyzerTutorialModal.steps.open.troubleshoot.bullets.path')}</li>
                </ul>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {sortedUnmappedJobs.map(job => {
                    const caption = stripMp4(basename(job.local_relative_path)) || t('common.notAvailable')
                    const isPendingDelete = pendingDeleteId === job.id
                    const isDeleting = bulkDeleting || deletingJobIds.has(job.id)
                    const deleteTitle = isPendingDelete
                        ? t('components.jobList.unmapped.delete.confirmTitle')
                        : t('components.jobList.unmapped.delete.title')
                    return (
                        <div key={job.id} className="flex flex-col items-start">
                            <div className="relative">
                                <JobThumbnail job={job} dirHandle={dirHandle} playable={!!job.local_relative_path} />
                                <Button
                                    type="button"
                                    variant="unstyled"
                                    size="none"
                                    className={`group absolute bottom-1 right-1 flex h-7 w-7 items-center justify-center rounded-full border p-0 leading-none shadow-sm transition ${
                                        isPendingDelete
                                            ? 'border-red-200 bg-red-50 text-red-600 hover:text-red-700'
                                            : 'border-white/60 bg-white/90 text-black hover:text-black'
                                    }`}
                                    title={deleteTitle}
                                    aria-label={deleteTitle}
                                    disabled={isDeleting}
                                    onClick={e => {
                                        e.stopPropagation()
                                        handleDeleteClick(job.id)
                                    }}
                                >
                                    <TrashIcon tone={isPendingDelete ? 'danger' : 'default'} />
                                    <span className="pointer-events-none absolute bottom-9 right-0 whitespace-nowrap rounded bg-black/80 px-2 py-1 text-[11px] text-white opacity-0 shadow-sm transition-opacity duration-150 group-hover:opacity-100">
                                        {deleteTitle}
                                    </span>
                                </Button>
                            </div>
                            <div className="mt-1 max-w-48 truncate text-xs text-gray-700" title={caption}>
                                {caption}
                            </div>
                        </div>
                    )
                })}
            </div>

            <DeleteAllModal
                open={showDeleteAllModal}
                count={unmappedJobs.length}
                bulkDeleting={bulkDeleting}
                onClose={() => setShowDeleteAllModal(false)}
                onConfirm={() => void handleDeleteAll()}
            />
        </div>
    )
}

export const JobList: React.FC<{
    jobs: JobSummary[]
    sortKey: JobListSortKey
    sortDir: JobListSortDir
    onToggleSort: (key: JobListSortKey) => void
    onOpen: (id: string) => void
    onDeleteJobs: (ids: string[]) => Promise<number>
    openingId?: string | null
    dirHandle?: FileSystemDirectoryHandle | null
    initialSyncComplete?: boolean
}> = ({
    jobs,
    sortKey,
    sortDir,
    onToggleSort,
    onOpen,
    onDeleteJobs,
    openingId,
    dirHandle = null,
    initialSyncComplete = false,
}) => {
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
                        <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                                trackEvent('joblist_folder_toggle')
                                toggleFolder(node.path)
                            }}
                            className="justify-start text-left py-1.5"
                            style={{ paddingLeft: depth * 12 }}
                            title={node.path}
                        >
                            <span className="w-4 text-slate-500">
                                {hasChildren || hasJobs
                                    ? isOpen
                                        ? t('components.jobList.folder.openIcon')
                                        : t('components.jobList.folder.closedIcon')
                                    : ''}
                            </span>
                            <span className="font-medium text-slate-900 flex items-center gap-2">
                                {node.name}
                                {isProcessing && (
                                    <span
                                        className="inline-flex items-center gap-1 text-[11px] text-blue-700"
                                        title={t('components.jobList.folder.inProgress', { count: node.activeJobs })}
                                    >
                                        <Spinner />
                                    </span>
                                )}
                            </span>
                            <span className="text-xs text-slate-500">({node.totalJobs})</span>
                            <div className="flex-1" />
                        </Button>
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
                                                        if (job.status === 'succeeded') onOpen(job.id)
                                                    }}
                                                    role="button"
                                                    tabIndex={0}
                                                    onKeyDown={e =>
                                                        job.status === 'succeeded' &&
                                                        (e.key === 'Enter' || e.key === ' ') &&
                                                        onOpen(job.id)
                                                    }
                                                >
                                                    <JobThumbnail
                                                        job={job}
                                                        dirHandle={dirHandle}
                                                        playable={!!job.local_relative_path}
                                                    />
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
            <div className="text-center text-gray-500 flex flex-col items-center gap-8">
                <div>
                    {initialSyncComplete ? t('components.jobList.waiting') : t('components.jobList.loading')}
                    <AnimatedDots />
                </div>
                {initialSyncComplete && (
                    <div className="mt-2 text-sm text-slate-500 max-w-md mx-auto">
                        {t('components.jobList.emptyTip')}
                    </div>
                )}
            </div>
        )
    }

    return (
        <div className="flex flex-col gap-3">
            <SortBar
                sortKey={sortKey}
                sortDir={sortDir}
                onToggleSort={onToggleSort}
                onExpandAll={expandAll}
                onCollapseAll={collapseAll}
            />
            {/* Unmapped jobs (no known local path) */}
            <UnmappedJobsSection
                unmappedJobs={unmappedJobs}
                sortKey={sortKey}
                sortDir={sortDir}
                dirHandle={dirHandle}
                onDeleteJobs={onDeleteJobs}
            />

            <div className="flex flex-col gap-1">{renderFolder(root, 0)}</div>
        </div>
    )
}
