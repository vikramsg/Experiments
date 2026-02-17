# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a minimal Python project.

- `main.py`: current executable entrypoint (`main()`).
- `pyproject.toml`: project metadata, Python version requirement, and dependencies.
- `plan.md`: implementation roadmap for LoRA fine-tuning work.
- `README.md`: project overview (expand this as features are added).
- `.env`: local environment variables (do not commit secrets).

As the codebase grows, prefer adding feature modules in a dedicated package directory and tests under `tests/`.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create and activate local virtual environment.
- `pip install -e .`: install the project in editable mode from `pyproject.toml`.
- `python main.py`: run the current entrypoint locally.
- `python -m pytest`: run tests (expected once test files are added).

If dependency management is moved to `uv` or another tool, document the exact replacement commands in `README.md`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Keep functions focused and side effects explicit.
- Add type hints for public functions and non-trivial internal APIs.
- Use clear module names that describe purpose (for example, `training_config.py`, `data_loader.py`).

No formatter/linter is configured yet; when adding one, standardize it in `pyproject.toml` and apply consistently.

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
  - sample output or logs for behavior changes.
