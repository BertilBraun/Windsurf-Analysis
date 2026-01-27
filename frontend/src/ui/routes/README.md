This folder contains the application's routing configuration and logic for authentication flows, user onboarding, and session management.

### Key Components

*   **Router**: The main entry point using `react-router-dom`. It defines the URL structure and includes automated page view tracking for analytics.
*   **AnalyzerRoute**: A protected route managing the authenticated user lifecycle:
    *   Forces login/signup if unauthenticated.
    *   Handles email verification states.
    *   Ensures a backend user record exists via `POST /users/{uid}`.
    *   Enforces legal consent (Terms/Privacy) via a `ConsentModal` before allowing access.
    *   Clears ephemeral demo sessions if a user attempts to access the full application.
*   **DemoRoute**: Manages ephemeral, anonymous sessions using Firebase `inMemoryPersistence` so sessions are wiped on refresh.

### Route Map

*   **Public Marketing Pages**: Wrapped in `AppShellLayout`.
    *   `/`: Home page.
    *   `/pricing`, `/faq`, `/technical`: Product information.
    *   `/terms`, `/privacy`, `/impressum`, `/contact`: Legal and contact information.
*   **Application Routes**:
    *   `/analyzer`: The primary authenticated tool.
    *   `/demo`: The anonymous sandbox environment.
*   **Redirects**:
    *   `/login` and `/signup` redirect to `/analyzer` to handle authentication state inline.
    *   Unknown paths redirect to the home page.

### Integration Details

*   **Firebase Auth**: Supports both anonymous (demo) and email/password (analyzer) authentication.
*   **Backend Synchronization**: Automatically ensures backend user records exist and tracks legal consent via `PATCH /users/{uid}/consent`.
*   **Analytics**: Uses `trackPageView` on every location change to monitor navigation.
