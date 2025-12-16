## Static frontend (Firebase Hosting)

### Preview locally
From repo root:

```bash
cd frontend
python -m http.server 8081
```

Then open `http://localhost:8081/`

### Deploy (Firebase Hosting)
1) Install Firebase CLI:

`npm i -g firebase-tools`

2) Login & deploy:

```bash
cd frontend
firebase login
firebase deploy --only hosting
```

Firebase will print your Hosting URL.
