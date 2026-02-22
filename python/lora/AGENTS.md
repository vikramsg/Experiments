# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a minimal Python project.

- `main.py`: current executable entrypoint (`main()`).
- `pyproject.toml`: project metadata, Python version requirement, and dependencies.
- `plan.md`: implementation roadmap for LoRA fine-tuning work.
- `README.md`: project overview (expand this as features are added).
- `.env`: local environment variables (do not commit secrets).

As the codebase grows, prefer adding feature modules in a dedicated package directory and tests under `tests/`.

## Training Workflow References
- `docs/training.md`: data requirements, evaluation plan, artifacts, and report template.

## Build, Test, and Development Commands
- Prefer using `just` tasks over direct `uv` commands when available.
- `just` (or `just --list`): list available tasks.
- `just venv`: create the local virtual environment with `uv`.
- `just sync`: install project dependencies with `uv`.
- `just run`: run the current entrypoint locally.
- `just test`: run tests with `pytest`.
- `just lint`: run Ruff lint checks.
- `just fix`: auto-fix Ruff lint issues.

### Experiment Management
**CRITICAL**: You MUST exclusively use the `justfile` for running and polling background experiments. Do not execute `uv run python main.py` directly for any background jobs.
- `just run-experiment <name> "<extra_args>"`: Run an experiment in the background (e.g., `just run-experiment exp_01 "--max-steps 1000"`).
- `just poll <name> [timeout=60]`: Poll the formatted `experiment.log` for a running experiment for a given number of seconds.
- `just poll-raw <name> [timeout=60]`: Poll the raw `nohup.out` terminal output.
- `just status <name>`: Check if an experiment is actively running.

Use `uv` strictly for environment management, dependency installation, and command execution within `just` commands. Do not use `pip` in this repository.

## Persistence

- When asked a question, do not just read 1 file or nothing and come back and give an answer.
- The answer must be backed by thorough research and citations. 
- Aim to have atleast 3 unique citations, research and then come back and answer.
- The objective is not to quickly start building. The objective is build the right thing.

## GitHub Repository Research
- When researching a GitHub repository's code or structure, clone it to `/tmp` and inspect files locally.
- Do not rely on web search or web browsing to walk repository contents.

## Artifact Conventions
- Capture adapter checkpoints, processor snapshots, and metrics reports for every training run.
- Record run summaries using the report template in `docs/training.md`.
- Ensure training, verification, and validation workflows emit verbose, step-by-step logs so progress is always visible.

## Expansion Guidance
- Place training config objects in `training_config.py`.
- Place dataset loading and preprocessing in `data_loader.py`.
- Place evaluation helpers in `evaluation.py`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Keep functions focused and side effects explicit.
- Add type hints for public functions and non-trivial internal APIs.
- Use clear module names that describe purpose (for example, `training_config.py`, `data_loader.py`).

Ruff is the standard linter for this repository. Configure linting rules only in `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.lint]`; avoid duplicating config in separate linter config files.

## Python Guidance
- Do not use `sys.path` mutations or `PYTHONPATH` overrides for imports.
- Keep imports at the top of the module; avoid inline imports unless strictly necessary.
- Do not use wildcard imports (`from module import *`); prefer explicit imports.
- Do not use `setattr` or `getattr` unless there is a strict, explicit requirement for dynamic attribute access.
- Avoid `if`/`else`, `try`/`except`, and similar branching/error-handling constructs unless they are strictly necessary for correctness.
- Use `dataclass` for structured data models; prefer `pydantic` models when validation or parsing is needed.
- Do not pass raw `dict` objects where a typed model is appropriate.
- Prefer fully typed code across modules, functions, and variables.
- Avoid `Any`; use precise types, `Protocol`, `TypedDict`, `TypeVar`, unions, or generics as appropriate.
- Prefix internal (non-public) functions with `_`.
- Keep functions small and single-purpose; avoid hidden side effects.
- Fail fast with specific exceptions; never use bare `except`. Do not write defensive fallback logic (e.g., silently returning empty values, inferring default behaviors on missing arguments, or ignoring missing state). Require explicit inputs, states, and configurations, and raise errors immediately if expectations are unmet.
- **CRITICAL**: ALWAYS use the project's structured `logging` setup. NEVER use raw `print()` statements. The logging setup allows always persisting logs to a file, which is a minimum requirement for all operations and traceability.
- Prefer immutable data models (`frozen=True` dataclasses or immutable pydantic models) unless mutation is required.
- Require explicit return type annotations for public and internal functions.
- Add Google-style docstrings for all public functions.

## Testing Guidelines
- Use `pytest` for new tests.
- Place tests in `tests/` with names like `test_<module>.py`.
- Name test cases by behavior (for example, `test_main_logs_greeting`).
- Prefer deterministic unit tests; do not use mocking. Instead, create actual data (even if fake) that is cleaned up after tests, such as by using the pytest `tmp_path` fixture.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style seen in history: `feat:`, `fix:`, `chore:`, optionally with scope (for example, `feat(zsh): ...`).
- Keep commits focused and atomic.
- PRs should include:
  - concise summary of what changed and why,
  - linked issue/task when available,
  - test evidence (command + result),
  - sample output for behavior changes.
