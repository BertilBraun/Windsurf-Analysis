## Firebase + Cloud Run setup (end-to-end)

This repo has:
- `frontend/`: React + TypeScript + Vite app deployed to **Firebase Hosting**
- `backend/`: FastAPI app deployed to **Google Cloud Run**
- `video_processing/`: Modal pipeline (orientation → detection → tracking) that reports results back to Cloud Run

This guide assumes **Windows** and uses the Firebase project id `gybelock-00` as an example.

---

## 0) Install tooling

### Node + Firebase CLI (for Hosting)
- Install Node.js (LTS)
- Install Firebase CLI:

```bash
npm i -g firebase-tools
firebase login
```

### Google Cloud SDK (for Cloud Run)
- Install Google Cloud SDK (`gcloud`)
- Authenticate:

```bash
gcloud auth login
gcloud auth application-default login
```

---

## 1) Create / select the Firebase project

1. Go to Firebase Console and create/select the project:
   - Project id: **`gybelock-00`**

2. In Cloud Console, ensure you’re using the same GCP project:

```bash
gcloud config set project gybelock-00
gcloud projects describe gybelock-00
```

---

## 2) Enable Firestore (IMPORTANT: use the default database)

Firebase client SDKs are simplest when you use the default Firestore database id:
- **`(default)`**

In Firebase Console:
- **Firestore Database → Create database**
- Choose **Native mode**
- Location: pick your region
- Finish setup

If you previously created a named database (e.g. `production`), keep it if you want, but for this repo’s frontend you should have a **`(default)`** database available.

---

## 3) Enable Authentication

Firebase Console → **Authentication → Sign-in method**:
- Enable **Email/Password**
- Enable **Google**

Optional but recommended:
- Authentication → Settings → Authorized domains: ensure your custom domain will be listed once connected.

---

## 4) Create the Firebase Web App (get config)

Firebase Console → **Project settings → Your apps → Add app → Web**

Copy the values from the “Firebase SDK snippet (Config)” and put them into:
- `frontend/.env.local` (local dev)
- `frontend/.env.production` (production build)

Example `frontend/.env.local`:

```env
VITE_BACKEND_URL=http://localhost:8080

VITE_FIREBASE_API_KEY=AIza...
VITE_FIREBASE_AUTH_DOMAIN=gybelock-00.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=gybelock-00
VITE_FIREBASE_STORAGE_BUCKET=gybelock-00.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=1:123456789:web:abcdef...
VITE_FIREBASE_DATABASE_ID=(default)
```

Example `frontend/.env.production`:

```env
VITE_BACKEND_URL=https://YOUR_CLOUD_RUN_URL

VITE_FIREBASE_API_KEY=...
VITE_FIREBASE_AUTH_DOMAIN=...
VITE_FIREBASE_PROJECT_ID=...
VITE_FIREBASE_STORAGE_BUCKET=...
VITE_FIREBASE_MESSAGING_SENDER_ID=...
VITE_FIREBASE_APP_ID=...
VITE_FIREBASE_DATABASE_ID=(default)
```

Notes:
- `VITE_*` variables are **public** (bundled into browser JS). Don’t put secrets in them.
- Vite loads env vars at **build time**; restart `npm run dev` after changes.

---

## 5) Run locally (dev loop)

### Backend (FastAPI)
From repo root:

```bash
pip install -r backend/requirements.txt
python backend/main.py
```

Backend should be at: `http://localhost:8080`

### Frontend (Vite)
From repo root:

```bash
cd frontend
npm install
npm run dev
```

Frontend should be at: `http://localhost:5173`

---

## 6) Deploy backend to Cloud Run

### 6.1 Enable APIs (one-time)

```bash
gcloud config set project gybelock-00
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### 6.2 Deploy
From repo root:

```bash
cd backend
gcloud run deploy backend --source . --region europe-west3 --allow-unauthenticated
```

Cloud Run will print your service URL, like:
- `https://backend-<project-number>.europe-west3.run.app`

Put that URL into:
- `frontend/.env.production` as `VITE_BACKEND_URL=...`

Then rebuild + redeploy Hosting (next section).

### 6.3 Configure required backend env vars

In Cloud Run → your service → “Edit & deploy new revision” → “Variables & secrets”:

- `FIREBASE_STORAGE_BUCKET=<your-bucket>` (often `<project>.appspot.com`)
- `MODAL_SHARED_SECRET=<random>` (same value you will set in Modal)
- `MODAL_TRIGGER_BASE_URL=<modal-trigger-url>` (set after deploying Modal)

Optional:

- `MAX_JOBS_PER_USER=5`
- `FIRESTORE_DATABASE=(default)`

### 6.4 Logs (runtime)
Cloud Run logs are under the **gybelock-00** project:

```bash
gcloud run services logs read backend --region europe-west3 --project gybelock-00 --limit 200
```

---

## 7) Deploy frontend to Firebase Hosting

From repo root:

```bash
cd frontend
npm run build
firebase use gybelock-00
firebase deploy --only hosting
```

You’ll get a URL like:
- `https://gybelock-00.web.app`

---

## 8) Connect custom domain `gybelock.de` (Squarespace DNS → Firebase Hosting)

Firebase Console → **Hosting → Add custom domain**:
- Add **`gybelock.de`**

Squarespace DNS:
- Add the exact records Firebase shows.
- **Important**: for Squarespace “Host” fields use **`@`** for the apex domain (not `gybelock.de`).

Common pattern:
- `A` record:
  - Host: `@`
  - Value: `199.36.158.100` (use exactly what Firebase shows)
- `TXT` verification:
  - Host: `@`
  - Value: `hosting-site=gybelock-00`
- `TXT` ACME challenge:
  - Host: `_acme-challenge`
  - Value: (token Firebase shows)

If Firebase certificate issuance fails with 403s to Squarespace IPs (`198.*`), it means DNS is still pointing at Squarespace. Verify with:

```bat
nslookup -type=A gybelock.de 8.8.8.8
nslookup -type=TXT _acme-challenge.gybelock.de 8.8.8.8
```

When `A` resolves to Firebase’s IP(s), click “Retry” in Firebase Hosting.

---

## 9) Deploy Modal (required for processing)

The backend triggers Modal, and Modal calls back into Cloud Run.

1. Install Modal and authenticate:

```bash
pip install modal
modal token new
```

2. Create a Modal secret named `backend-secret` containing:

- `MODAL_SHARED_SECRET` (must match your backend env var)
- `CLOUD_RUN_BASE_URL` (your Cloud Run backend base URL)

3. Deploy:

```bash
python video_processing/deploy.py
```

4. Copy the trigger endpoint URL Modal prints and set:

- `MODAL_TRIGGER_BASE_URL=<that-url>`

---

## 10) Common troubleshooting

### Browser CORS errors calling Cloud Run
CORS headers come from the FastAPI app. If the revision is crashing, you’ll see “CORS blocked” even though it’s actually a backend failure.
- Check Cloud Run logs (section 6.4)
- Ensure the backend is returning responses for `OPTIONS` and your allowed origins include your Hosting domain.
