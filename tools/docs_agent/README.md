# Docs Agent (hash-based)

This folder contains a small, provider-agnostic “docs maintenance” toolchain.

Goal: only (re)process files/folders whose content changed, so you can run it periodically (locally or in CI) without re-documenting the entire repo every time.

## What exists

- `tools/docs_agent/run.py`: fast inventory + hashing cache
- `tools/docs_agent/generate.py`: optional docs generator (folder `README.md` + inline docs)
- `tools/docs_agent/worktree_run.py`: run generation in a clean detached worktree

## Environment (.env)

This repo ignores `.env` files. Create a root `.env` (or copy from `.env.example`) and put secrets there.

Variables supported:

- Gemini (OpenAI-compatible endpoint):
  - `GEMINI_API_KEY=...`
  - `DOCS_AGENT_GEMINI_MODEL=gemini-3-flash-preview`
  - `DOCS_AGENT_GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`
- OpenAI (optional):
  - `OPENAI_API_KEY=...`
  - `DOCS_AGENT_OPENAI_MODEL=...`
  - `OPENAI_BASE_URL=https://api.openai.com/v1`

If you want to load a different env file path, set `DOCS_AGENT_ENV_PATH`.

## Inventory

From the repo root:

```powershell
python tools\docs_agent\run.py --write-state
python tools\docs_agent\run.py --changed-only
python tools\docs_agent\run.py --print-json
```

Notes:
- Only **tracked** files are scanned (`git ls-files`), so vendor/untracked output folders are naturally ignored.
- State is stored in `.docs_agent/state.json` (and `.docs_agent/**` is ignored via `.gitignore`).

## Generating docs

### Folder README stubs (no LLM)

```powershell
python tools\docs_agent\generate.py --update-readmes --apply --write-state
```

### With Gemini 3 Flash (inline docs + folder READMEs)

```powershell
python tools\docs_agent\generate.py --llm gemini --update-inline-docs --update-readmes --force --max-files 10 --max-folders 10 --apply --write-state --format-python
```

### With OpenAI (optional)

```powershell
python tools\docs_agent\generate.py --llm openai --update-inline-docs --update-readmes --max-files 10 --max-folders 10 --apply --write-state --format-python
```

## Running from a clean worktree

```powershell
python tools\docs_agent\worktree_run.py -- --llm gemini --update-inline-docs --update-readmes --force --max-files 10 --max-folders 10 --apply
```

This generates changes in a detached worktree, writes a patch to `.docs_agent/last.patch`, applies it to your current repo, then updates `.docs_agent/state.json`.
