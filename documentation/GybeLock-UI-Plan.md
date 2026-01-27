# GybeLock Style Guideline + UI Plan

Status: **design document / not necessarily implemented**.

This is a UI/brand plan for the GybeLock frontend. Some file paths below were written during a refactor and may
need adjustment; treat them as guidance, not a guarantee.

## Brand & tone
- **Name**: GybeLock
- **Concept**: “Locked onto the jibe” — confident, technical, and inviting.
- **Voice**: short, calm, instructional. Prefer action verbs (“Analyze”, “Open”, “Upload”) over jargon.

## Visual identity
- **Logo idea**: wordmark “GybeLock” + simple mark that reads as a **jibe arc** (sweeping curve) that “locks” (small notch/corner) onto a point.
- **Usage**:
  - Header: mark + wordmark.
  - Player mode (modal): **mark + wordmark only**, clicking returns to Home (closes modal).
  - Clear space: at least the height of the “G” around the mark.

## Color system (light UI, turquoise accents)
Define tokens as CSS variables (so Tailwind + inline styles can share them).
- **Background**: white / near-white
- **Surfaces**: subtle gray-tinted white
- **Text**: slate/charcoal
- **Accent (turquoise)**:
  - Primary: turquoise 600-ish (buttons, active states)
  - Hover: turquoise 700-ish
  - Soft tint: turquoise 50-ish (badges, focus backgrounds)
- **Status**:
  - Success: green
  - Warning: amber
  - Danger: red
  - Info: turquoise/cyan

Suggested palette (can be adjusted after you see it in-app):
- `--brand-600`: #0D9488
- `--brand-700`: #0F766E
- `--brand-50`:  #F0FDFA
- `--text`: #0F172A
- `--muted`: #475569
- `--border`: #E2E8F0
- `--surface`: #F8FAFC
- `--shadow`: use Tailwind shadows, avoid heavy drop shadows.

## Typography
- **Font**: Inter (already used). Keep system fallback.
- **Type scale** (Tailwind mapping):
  - Page title: `text-2xl font-semibold`
  - Section title: `text-lg font-semibold`
  - Body: `text-sm leading-6`
  - Muted/help: `text-xs text-slate-500`

## Layout & navigation
- **Global frame (all pages except player-mode)**:
  - Top header bar: left = logo “GybeLock”, right = nav links `Home`, `Analyzer`, `FAQ`.
  - Content container: max width (e.g. ~1200–1400px), generous whitespace.
  - Footer: `Terms of Use`, `Privacy Policy`, `Product Reviews`, `Contact`.
- **Player-mode frame (when `PlayerModal` is open)**:
  - Hide global nav.
  - Show **logo-only** affordance (click returns to Home / closes modal).

## Page specs
### Home
- Hero: “Analyze your jibes. Improve faster.”
- 3-step “How it works”: Upload → Process → Review.
- Short “What GybeLock measures” section (bullets).
- CTA button: “Open Analyzer”.

### FAQ
- Simple accordion list: supported formats, where videos live (ingress folder), privacy, troubleshooting.

### Analyzer
- **Primary element**: processed videos **grid** (current `JobList` already uses a responsive grid).
- Controls row: sort, filter (optional later), status chips.
- **Ingress**: move from top-of-page panel to a **floating bottom-right widget**:
  - Collapsed: small pill/FAB with icon + optional badge count; if uploading, show **circular progress ring**.
  - Expanded: directory selection + status line + upload list (reuse current `IngressPanel` content).

## Component guidelines (key)
- **Buttons**: introduce variants `primary` (brand), `secondary` (neutral), `danger`.
- **Cards/tiles** (video thumbnails): light surface, subtle border, hover lift (small shadow), clear status labeling.
- **Modal**: default should match light theme. For Player, full-bleed content with minimal chrome.
- **Focus states**: always visible (`ring-2 ring-[--brand-600] ring-offset-2`).
- **Accessibility**: maintain contrast (avoid light turquoise text), keyboard operability for grid items.

## Implementation plan (grounded in current code)

### 1) Add an App shell (header/footer) and simple nav state
- Create `AppShell` wrapping content with header + footer.
- Update routing in `frontend/src/ui/routes/Router.tsx` to include `home/analyzer/faq` post-auth.
- Replace the inline margin wrapper in `frontend/src/ui/App.tsx` with `AppShell` so margins and typography are centralized.

### 2) Add pages: Home + FAQ, refactor MainPage → Analyzer
- Create `frontend/src/ui/screens/HomePage.tsx`.
- Create `frontend/src/ui/screens/FaqPage.tsx`.
- Refactor `frontend/src/ui/screens/MainPage.tsx` into `AnalyzerPage` layout: grid prominent, controls row, remove the big `IngressPanel` from the top.

### 3) Floating Ingress widget
- Build `frontend/src/ui/components/IngressWidget.tsx` that uses `useIngressScanner` (same logic as `IngressPanel`).
- Reuse visual sub-parts from `frontend/src/ui/components/IngressPanel.tsx` but in a collapsible container anchored bottom-right.

### 4) Light-theme primitives (Modal/Button) and player-mode header
- Update `frontend/src/ui/components/Modal.tsx` so default modal chrome is light (it’s currently dark by default).
- Update `frontend/src/ui/components/Button.tsx` to support variants + consistent sizing.
- Update `frontend/src/ui/components/PlayerModal.tsx` to render “player-mode” header (logo-only + close/back behavior) while keeping the current modal flow.

### 5) Design tokens
- Define CSS variables and base typography in `frontend/src/index.css`.
- Extend Tailwind colors to reference CSS vars in `frontend/tailwind.config.js`.

### 6) Footer links
- Add footer component with the four links; route them to simple placeholder pages or `mailto:` for Contact.

## Open items (decide during implementation)
- Whether to add an icon set (e.g. `lucide-react`) or keep inline SVG only.
- Whether footer links are real routes vs external links (legal pages often start as placeholders).

