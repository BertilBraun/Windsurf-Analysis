## Frontend (React + TypeScript + Vite) + Firebase Hosting

For end-to-end setup (Firebase + Cloud Run + Modal), see `documentation/DEPLOYMENT.md`.

### Setup (Firebase Console)
- Create / select your Firebase project
- **Authentication**: enable a provider (e.g. Google)
- **Firestore Database**: create the database (Native mode)
- Copy your web app config into env vars (see below)

### Configure env vars
This app reads config from Vite env vars (`VITE_*`).

Create a local file **(don’t commit it)**:
- `frontend/.env.local`

Use `frontend/env.example` as a template.

### Debug locally (hot reload / watch mode)
From repo root:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

#### Point the frontend at your local backend
Set this in `frontend/.env.local`:
- `VITE_BACKEND_URL=http://localhost:8080`

Then restart `npm run dev`.

### Debug locally as “built” app
From repo root:

```bash
cd frontend
npm run build
npm run preview -- --host --port 5173
```

### Debug locally via Firebase Hosting emulator (optional)
This serves what would be deployed (from `dist/`), so build first:

```bash
cd frontend
npm run build
firebase emulators:start --only hosting
```

### Deploy (Firebase Hosting)
```bash
cd frontend
npm run build
firebase deploy --only hosting
```

### Notes
- The frontend signs in with Firebase Auth, gets an ID token, and calls the backend with `Authorization: Bearer <token>`.
