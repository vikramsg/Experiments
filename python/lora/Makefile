
.PHONY: help venv sync run test lint fix

help:
	@echo "Usage:"
	@echo "  make <target>"
	@echo ""
	@echo "Targets:"
	@printf "  %-10s %s\n" "help" "Show this help message"
	@printf "  %-10s %s\n" "venv" "Create virtual environment with uv"
	@printf "  %-10s %s\n" "sync" "Install dependencies from uv.lock"
	@printf "  %-10s %s\n" "run" "Run the main entrypoint"
	@printf "  %-10s %s\n" "test" "Run pytest"
	@printf "  %-10s %s\n" "lint" "Run Ruff lint checks"
	@printf "  %-10s %s\n" "fix" "Auto-fix Ruff lint issues"

venv:
	uv venv

sync:
	uv sync

run:
	uv run python main.py

test:
	uv run pytest

lint:
	uv run ruff check .

fix:
	uv run ruff check . --fix
