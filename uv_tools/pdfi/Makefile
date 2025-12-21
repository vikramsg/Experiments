.PHONY: test build tool-install

test:
	uv run pytest

build:
	rm -rf dist
	uv run python -m build

tool-install:
	uv tool install --editable .
