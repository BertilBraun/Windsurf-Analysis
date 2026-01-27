# Docs Agent

Automated documentation maintenance toolchain using LLMs (Gemini or OpenAI) with hash-based incremental updates.

## Core Tools

*   **run.py**: Performs repository inventory and change detection using SHA256 hashes.
*   **generate.py**: Main engine for updating inline docstrings/JSDoc and generating folder-level README.md files.
*   **worktree_run.py**: Executes the generation process in a detached Git worktree to produce a clean patch and avoid local workspace conflicts.
*   **verify.py**: Provides AST-based safety checks for Python files to ensure LLM updates only modify docstrings and not runtime logic.

## Key Features

*   **Incremental Processing**: Uses a local state file (`.docs_agent/state.json`) to skip unchanged files and folders.
*   **LLM Caching**: Persists LLM responses to disk (`.docs_agent/llm_cache/`) keyed by model, prompt version, and input content.
*   **Context-Aware READMEs**: Generates folder documentation using the contents of local files and the READMEs of immediate subfolders (processed bottom-up).
*   **Public API Focus**: Prompts LLMs to only document exported TypeScript symbols and public Python members (skipping private/internal functions).
*   **Safety Verification**: Uses Python's `ast` module to compare code before and after LLM processing, rejecting any changes that modify runtime behavior.
*   **Rate Limit Management**: Includes configurable sleep intervals and exponential backoff retries for 429/503 errors.
*   **Python Formatting**: Optional integration with `ruff` to format updated Python files.

## Configuration

*   **Environment**: Loads settings from `.env` or a path specified by `DOCS_AGENT_ENV_PATH`.
*   **API Keys**: Requires `GEMINI_API_KEY` or `OPENAI_API_KEY`.
*   **Model Selection**: Configurable via `DOCS_AGENT_GEMINI_MODEL` or `DOCS_AGENT_OPENAI_MODEL`.
*   **File Support**: Defaults to `.py`, `.ts`, and `.tsx`; additional extensions can be added via CLI flags.
*   **Storage Paths**: State and cache locations can be overridden via `DOCS_AGENT_STATE_PATH` and `DOCS_AGENT_CACHE_DIR`.

## TODO

*   Implement AST-based verification for TypeScript/JavaScript files.
*   Add support for more LLM providers beyond Gemini and OpenAI.
*   Expand language-specific public API detection for other file types.
