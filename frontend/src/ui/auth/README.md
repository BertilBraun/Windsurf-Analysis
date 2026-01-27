Authentication logic and state management using Firebase Auth.

### Core Components
* **AuthProvider**: React context provider managing the Firebase authentication lifecycle, user state, and token refreshing.
* **useAuth**: Hook to access authentication state and methods (login, signup, logout, etc.).

### Features
* **Authentication Methods**: Supports Email/Password and Google OAuth (popup).
* **Email Verification**: Workflows for sending verification emails and checking status; `isAuthenticated` is only true for verified users.
* **Backend Synchronization**: Automatically ensures a user record exists in the backend via `POST /users/{uid}` after signup or social login.
* **Authorized Fetch**: A `fetch` wrapper that:
    * Injects the Firebase ID token as a `Bearer` header.
    * Prefixes requests with the configured `API_BASE`.
    * Handles 401 Unauthorized responses by signing out and redirecting to `/demo` or `/analyzer`.
* **Analytics Integration**: Tracks `auth_login`, `auth_signup`, and `auth_logout` events; synchronizes user ID with the analytics utility.
* **Session Management**: Restores auth state on page load and clears legacy basic auth credentials.

### State Properties
* `user`: Raw Firebase User object.
* `isAuthReady`: True once Firebase has finished the initial state check.
* `isSignedIn`: True if a user is logged in.
* `isAuthenticated`: True if the user is logged in and email is verified.
* `needsEmailVerification`: True if logged in but email is not yet verified.
* `authHeader`: The current `Bearer <token>` string.
* `uid` / `email`: Convenience accessors for current user details.
