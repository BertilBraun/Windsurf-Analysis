# Tools

Utility scripts and automation toolchains for repository maintenance and development.

## Subfolders

*   **docs_agent**: Automated documentation maintenance toolchain using LLMs (Gemini or OpenAI).
    *   **Incremental Updates**: Uses SHA256 hashes and a local state file to skip unchanged content.
    *   **Docstring Generation**: Updates inline docstrings/JSDoc for public APIs in Python and TypeScript.
    *   **README Generation**: Bottom-up generation of folder-level READMEs using local file context and subfolder documentation.
    *   **Safety & Verification**: AST-based checks for Python to prevent logic changes; optional `ruff` formatting.
    *   **Execution**: Supports running in detached Git worktrees to avoid local workspace conflicts.
    *   **Caching**: Persists LLM responses to disk to reduce API costs and latency.

## TODO

*   Add common development scripts (e.g., linting, formatting, or deployment helpers).
*   Implement AST-based verification for TypeScript/JavaScript files in `docs_agent`.
*   Expand `docs_agent` support for additional LLM providers and file types.
