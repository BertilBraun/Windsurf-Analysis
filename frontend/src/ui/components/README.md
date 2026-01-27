Reusable UI components for the application, ranging from core design system elements to feature-specific widgets.

### Core UI Elements
*   **Button**: Reusable button supporting multiple variants (primary, ghost, danger, outline, etc.), sizes, and a pending (loading) state with a spinner.
*   **Modal**: Base overlay component with support for headers, backdrops, keyboard escape handling, and focus management.
*   **Typography**: Standardized `Heading`, `Text`, `TextStack`, and `TextStrong` components for consistent font styling, hierarchy, and spacing.
*   **Spinner / AnimatedDots**: Visual loading indicators; `AnimatedDots` maintains a fixed width to prevent layout shifts.
*   **LogoButton**: Brand logo wrapper that functions as either a router `Link` or a standard `Button`.

### Layout & Navigation
*   **AppShell**: The primary layout wrapper containing the sticky header, navigation menu, main content area (`Outlet`), and footer.
*   **LanguageSwitcher**: A dropdown menu for switching between supported languages (EN, DE, ES, IT) with flag icons.
*   **AnalyticsConsentBanner**: A bottom-fixed banner for tracking consent that manages a `--analytics-consent-offset` CSS variable for layout adjustments.
*   **UnsupportedBrowserBanner**: Displays a warning if the browser lacks required APIs like WebCodecs or File System Access.

### Analyzer & Video Management
*   **IngressWidget**: A floating status widget that monitors a local directory for new videos, manages background uploads, and handles directory permissions.
*   **JobList**: Renders a hierarchical, folder-based view of video processing jobs with sorting, folder expansion/collapse, and a section for "unmapped" jobs (those without local paths).
*   **JobThumbnail**: Generates and caches video thumbnails locally using `mediabunny` and IndexedDB; displays processing status badges (e.g., stabilization, tracking) for active jobs.
*   **PlayerModal**: A full-screen wrapper for the video player, integrating drawing tools, stabilization toggles, and issue reporting.

### Feature Modals & Sections
*   **AnalyzerTutorialModal**: A multi-step walkthrough guiding users through folder selection and the analysis workflow, including a link to a video walkthrough.
*   **ConsentModal**: A mandatory dialog for accepting terms of service and optional marketing communications.
*   **FeedbackModal / HelpModal**: Interfaces for submitting feedback to the backend or viewing support contact information.
*   **SettingsModal**: Provides user account management, including language selection, logout, and account deletion.
*   **KeyboardShortcutsModal**: Displays a reference list of player control shortcuts (e.g., seek, play/pause, speed).
*   **GetStartedSection / SupportProjectSection**: Call-to-action sections for the landing page and project donations (PayPal).

### TODO
*   Implement i18n for specific processing labels in `IngressWidget` (e.g., "Processing X/Y files").
*   Re-enable or refine the `UploadQuality` selector in `SettingsModal`.
