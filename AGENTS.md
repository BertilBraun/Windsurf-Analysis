<INSTRUCTIONS>
Follow these rules for any work done in this repository.

## Instruction precedence
- System > Developer > User > Repo instructions (`AGENTS.md`).
- If a nested `AGENTS.md` exists under a subfolder, it overrides this file for files in that subtree.

## Code style (repo-wide)
- Keep files, classes, and functions short and focused (single responsibility).
- Avoid deep nesting; prefer guard clauses, early returns, and extracting helper functions.
- Structure code into sensible modules; keep boundaries clear.
- Don't repeat yourself more than twice; extract shared logic when a 3rd copy would appear or earlier.
- Naming: ALWAYS use descriptive variable/function names; NEVER abbreviate.
  - Prefer `player` over `p`, `indices` over `idxs`, `config` over `cfg`.
- Prefer straightforward control flow and simple data structures over cleverness.
- When touching code, keep diffs minimal; avoid drive-by refactors.
- Use strong typing - ALWAYS. Always use dataclasses and typedefinitions over generic objects or maps. Use enums over string constants.

## Safety and hygiene
- Never commit or print secrets. Treat `.env` as sensitive; use `.env.example` for examples.
- Avoid editing generated/vendor directories (`frontend/dist/`, `frontend/node_modules/`, `.docs_agent/`, `__pycache__/`) unless explicitly asked.
- If you change runtime behavior, call it out in the final summary and include how it was verified.
- Concurrency safety: DO NOT remove or revert code that appeared “in the meantime” (e.g., manual edits or another agent’s changes).

## Repo map
- `backend/`: FastAPI backend for GybeLock (Firebase + API).
- `frontend/`: React + Vite frontend (Firebase hosting).
- `video_processing/`, `train/`: Python CV pipeline + training tooling.
- `tools/docs_agent/`: Local "docs agent" scripts for generating/updating documentation.
- `documentation/`: Human docs; start at `documentation/README.md`.

## Validation (pick the closest fit)
- Python sanity: `python -m compileall backend tools video_processing train`
- Frontend run: `cd frontend; npm run dev`

## Repo skills (installed in your Codex home)
- Use `$plan` when you want a tracked, step-by-step execution plan using the `update_plan` tool.
- Use `$review` when you want a structured code review (risk, correctness, tests, style, follow-ups).
- If new skills aren't listed yet, restart the Codex session so it re-discovers skills.
</INSTRUCTIONS>
