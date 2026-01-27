This folder contains the top-level page components representing the primary routes and user interfaces of the application.

### Core Application Screens
*   AnalyzerPage: The primary dashboard for authenticated users. Manages the job list, handles local ingress folder selection via the File System Access API, and integrates the video player for reviewing results.
*   DemoPage: A simplified interface for single-video analysis. Supports local file uploads with SHA-256 hashing, sample video testing, and progress tracking for guest users.
*   HomePage: The main landing page featuring a hero section with video comparisons, problem/solution overviews, and a step-by-step guide to the workflow.

### Authentication
*   LoginPage: Handles user sign-in via email/password or Google OAuth. Includes functionality for password recovery and navigation to signup.
*   SignupPage: Manages new user registration, including mandatory legal consent for terms and optional marketing preferences.

### Information & Legal
*   FaqPage: Provides a list of collapsible frequently asked questions and troubleshooting information for common video processing issues.
*   PricingPage: Details the current free beta phase and the planned transition to a pay-per-use model without subscriptions.
*   LegalPage: A multi-purpose component rendering Terms of Use, Privacy Policy, Impressum, and Contact information. It also hosts the Google Analytics consent management UI.
*   TechnicalPage: Fetches and renders the project's TECHNICAL.md file using a custom Markdown renderer and provides links to the GitHub repository.

### TODO
*   Implement robust error boundaries for individual page components to prevent full-app crashes.
*   Add breadcrumb navigation for deep-linked informational and legal pages.
*   Optimize the loading state and transitions between the Demo and Analyzer workflows.
