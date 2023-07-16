#!/usr/bin/env bash

set -o verbose

poetry run mypy . --exclude tests/
poetry run isort .
poetry run black .
# poetry run ruff . --fix