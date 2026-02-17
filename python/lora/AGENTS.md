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
- Prefer using `make` targets over direct `uv` commands when available.
- `uv run python main.py`: run the current entrypoint locally.
- `make help`: list available Make targets.
- `make venv`: create the local virtual environment with `uv`.
- `make sync`: install project dependencies with `uv`.
- `make run`: run the current entrypoint.
- `make test`: run tests with `pytest`.
- `make lint`: run Ruff lint checks.

Use `uv` strictly for environment management, dependency installation, and command execution. Do not use `pip` in this repository.

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

## Testing Guidelines
- Use `pytest` for new tests.
- Place tests in `tests/` with names like `test_<module>.py`.
- Name test cases by behavior (for example, `test_main_prints_greeting`).
- Prefer deterministic unit tests; isolate network/model-download steps behind mocks when practical.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style seen in history: `feat:`, `fix:`, `chore:`, optionally with scope (for example, `feat(zsh): ...`).
- Keep commits focused and atomic.
- PRs should include:
  - concise summary of what changed and why,
  - linked issue/task when available,
  - test evidence (command + result),
  - sample output for behavior changes.
