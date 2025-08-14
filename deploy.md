# Deploy

## Install Modal

```bash
pip install modal
python -m modal setup
```

## Build frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

## Deploy

```bash
cd ../.. # back to Windsurf Analysis root
modal deploy deploy_server.py
modal deploy deploy_inference.py
```
