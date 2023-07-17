.PHONY: install test lint check run 

help:
	@echo "install - install dependencies with poetry"
	@echo "test - run unit tests"
	@echo "lint - run linter and checks"
	@echo "check - run static checks"
	@echo "run - run game"

test:
	poetry run pytest -vv test

check:
	./static_checks.sh

lint:
	./linter.sh
	
install:
	poetry install --no-root
	poetry shell

run:
	poetry run python -m src.tic_tac_toe 