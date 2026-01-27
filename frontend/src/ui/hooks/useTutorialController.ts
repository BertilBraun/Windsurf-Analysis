/**
 * @fileoverview Provides the logic for managing the analyzer tutorial flow,
 * including progress tracking, persistence, and automatic triggers.
 */

import React from 'react'
import { loadSetting, saveSetting } from '../utils/idb'
import { trackEvent } from '../utils/analytics'
import type { JobDetail } from '../types'
import type { AnalyzerTutorialModalProps } from '../components/AnalyzerTutorialModal'

const ANALYZER_TUTORIAL_SEEN_KEY = 'analyzerTutorialSeen.v2'
const ANALYZER_TUTORIAL_PROGRESS_KEY = 'analyzerTutorialProgress.v2'

type TutorialOpenSource = 'header' | 'empty_state' | 'auto'

type TutorialStepKey = 'what' | 'watch-folder' | 'add-videos' | 'review-riding' | 'feedback-reports'

const ALL_TUTORIAL_STEP_KEYS: TutorialStepKey[] = [
    'what',
    'watch-folder',
    'add-videos',
    'review-riding',
    'feedback-reports',
]
const ONBOARDING_TUTORIAL_STEP_KEYS: TutorialStepKey[] = ['what', 'watch-folder', 'add-videos']
const PLAYER_TUTORIAL_STEP_KEYS: TutorialStepKey[] = ['review-riding', 'feedback-reports']

/**
 * Configuration options for the tutorial controller.
 */
type TutorialControllerOptions = {
    /** Current directory handle for video ingress. */
    dirHandle: FileSystemDirectoryHandle | null
    /** Currently selected job, used to trigger contextual tutorial steps. */
    selectedJob: JobDetail | null
    /** Callback to trigger the folder selection process. */
    onPickIngressFolder: () => void
}

/**
 * The state and control functions exposed by the tutorial controller.
 */
type TutorialControllerState = {
    /** Whether the tutorial modal is currently visible. */
    showTutorial: boolean
    /** Function to manually open the tutorial with specific steps or starting point. */
    openTutorial: (source: TutorialOpenSource, stepKeys?: TutorialStepKey[] | null, startAt?: TutorialStepKey) => void
    /** Props to be passed directly to the AnalyzerTutorialModal component. */
    tutorialModalProps: AnalyzerTutorialModalProps
}

/**
 * A hook that manages the tutorial state, tracking progress in IndexedDB
 * and handling automatic triggers for onboarding and player-specific steps.
 *
 * @param options - Configuration for folder selection and job context.
 * @returns The current tutorial state and methods to control the tutorial UI.
 */
export const useTutorialController = ({
    dirHandle,
    selectedJob,
    onPickIngressFolder,
}: TutorialControllerOptions): TutorialControllerState => {
    const [showTutorial, setShowTutorial] = React.useState<boolean>(false)
    const [tutorialStepIndex, setTutorialStepIndex] = React.useState<number>(0)
    const [tutorialStepKeys, setTutorialStepKeys] = React.useState<TutorialStepKey[] | null>(null)
    const [tutorialSeenSteps, setTutorialSeenSteps] = React.useState<Set<TutorialStepKey>>(() => new Set())
    const [tutorialProgressLoaded, setTutorialProgressLoaded] = React.useState<boolean>(false)

    React.useEffect(() => {
        loadSetting<TutorialStepKey[] | null>(ANALYZER_TUTORIAL_PROGRESS_KEY).then(saved => {
            const valid = Array.isArray(saved) ? saved.filter(key => ALL_TUTORIAL_STEP_KEYS.includes(key)) : []
            setTutorialSeenSteps(prev => {
                const merged = new Set(prev)
                for (const key of valid) merged.add(key)
                return merged
            })
            setTutorialProgressLoaded(true)
        })
    }, [])

    const markTutorialStepsSeen = React.useCallback((keys: TutorialStepKey[]) => {
        setTutorialSeenSteps(prev => {
            const next = new Set(prev)
            for (const key of keys) next.add(key)
            void saveSetting(ANALYZER_TUTORIAL_PROGRESS_KEY, Array.from(next))
            return next
        })
    }, [])

    const getContextualStepKeys = React.useCallback(
        (extraKeys: TutorialStepKey[]) => {
            const merged = new Set(tutorialSeenSteps)
            for (const key of extraKeys) merged.add(key)
            return ALL_TUTORIAL_STEP_KEYS.filter(key => merged.has(key))
        },
        [tutorialSeenSteps]
    )

    const openTutorial = React.useCallback(
        (source: TutorialOpenSource, stepKeys?: TutorialStepKey[] | null, startAt?: TutorialStepKey) => {
            trackEvent('open_tutorial', { source })
            const keysToUse = stepKeys ?? ALL_TUTORIAL_STEP_KEYS
            const startIdx = startAt ? Math.max(0, keysToUse.indexOf(startAt)) : 0
            setTutorialStepKeys(stepKeys ?? null)
            setTutorialStepIndex(startIdx >= 0 ? startIdx : 0)
            setShowTutorial(true)
            markTutorialStepsSeen(keysToUse)
        },
        [markTutorialStepsSeen]
    )

    const closeTutorial = React.useCallback(() => {
        trackEvent('close_tutorial')
        setShowTutorial(false)
        setTutorialStepKeys(null)
        void saveSetting(ANALYZER_TUTORIAL_SEEN_KEY, true)
    }, [])

    // Auto-open the tutorial once for new users (no ingress folder and no jobs yet).
    React.useEffect(() => {
        loadSetting<boolean>(ANALYZER_TUTORIAL_SEEN_KEY).then(seen => {
            if (seen) return
            openTutorial('auto', ONBOARDING_TUTORIAL_STEP_KEYS, 'what')
        })
    }, [openTutorial])

    const hasOpenedPlayerRef = React.useRef(false)

    React.useEffect(() => {
        if (!tutorialProgressLoaded) return
        if (showTutorial) return
        if (!selectedJob) return
        if (hasOpenedPlayerRef.current) return
        hasOpenedPlayerRef.current = true
        if (
            tutorialSeenSteps.has('review-riding') &&
            tutorialSeenSteps.has('feedback-reports')
        )
            return
        const stepKeys = getContextualStepKeys(PLAYER_TUTORIAL_STEP_KEYS)
        openTutorial('auto', stepKeys, 'review-riding')
    }, [getContextualStepKeys, openTutorial, selectedJob, showTutorial, tutorialProgressLoaded, tutorialSeenSteps])

    const tutorialModalProps: AnalyzerTutorialModalProps = React.useMemo(
        () => ({
            onClose: closeTutorial,
            stepIndex: tutorialStepIndex,
            onStepIndexChange: setTutorialStepIndex,
            onPickIngressFolder,
            ingressFolderName: dirHandle?.name ?? null,
            stepKeys: tutorialStepKeys,
        }),
        [closeTutorial, dirHandle?.name, onPickIngressFolder, tutorialStepIndex, tutorialStepKeys]
    )

    return {
        showTutorial,
        openTutorial,
        tutorialModalProps,
    }
}
