# Docs Agent

This repo includes a small “docs maintenance” tool under `tools/docs_agent/` that can:

- Track per-file hashes (TS/TSX + Python) so only changed files are reprocessed.
- Optionally generate/update:
  - per-folder `README.md`
  - inline docs (Python docstrings, TS/TSX JSDoc) via an LLM provider

## Quick start

Inventory and write/update the hash cache:

```powershell
python tools\docs_agent\run.py --write-state
python tools\docs_agent\run.py --changed-only
```

Generate folder README stubs (no LLM):

```powershell
python tools\docs_agent\generate.py --update-readmes --apply --write-state
```

Create `.env` from `.env.example` and set your key(s).

Generate inline docs + folder READMEs with Gemini:

```powershell
python tools\docs_agent\generate.py --llm gemini --update-inline-docs --update-readmes --force --max-files 10 --max-folders 10 --apply --write-state --format-python
```

Generate inline docs + folder READMEs with OpenAI (optional):

```powershell
python tools\docs_agent\generate.py --llm openai --update-inline-docs --update-readmes --max-files 10 --max-folders 10 --apply --write-state --format-python
```

## Safer “clean repo” runs

If you want generation to happen in a detached clean worktree (so your working directory stays untouched while generating), use:

```powershell
python tools\docs_agent\worktree_run.py -- --llm openai --update-inline-docs --update-readmes --max-files 10 --max-folders 10 --apply
```

For Gemini in a clean worktree:

```powershell
python tools\docs_agent\worktree_run.py -- --llm gemini --update-inline-docs --update-readmes --force --max-files 10 --max-folders 10 --apply
```

## Periodic runs

- Local (Windows Task Scheduler): schedule a weekly task that runs one of the commands above from the repo root.
- CI: run on a schedule on a clean checkout and open a PR with the resulting changes (recommended).
