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
# Deploy inference backend first
cd inference
modal deploy inference.py
cd ..

# Deploy main app
cd .. # back to Windsurf Analysis root
modal deploy main.py
```